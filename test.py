"""
Stress benchmark for Mooncake Master RPC service.

Tests multiple clients concurrently registering with the master and performing
RPC operations (health_check, is_exist, batch_is_exist, put, get, remove,
get_replica_desc) to measure master RPC throughput and latency under load.

Usage:
    # Single client, default operations
    python stress_master_rpc_benchmark.py \
        --master-server localhost:50051 \
        --metadata-server http://127.0.0.1:8080/metadata \
        --num-clients 1

    # 8 clients concurrent stress test
    python stress_master_rpc_benchmark.py \
        --master-server localhost:50051 \
        --metadata-server http://127.0.0.1:8080/metadata \
        --num-clients 8 \
        --ops-per-client 500 \
        --rpc-types health_check,is_exist,put_get,batch_is_exist

    # Full stress with detailed per-client stats
    python stress_master_rpc_benchmark.py \
        --master-server localhost:50051 \
        --metadata-server http://127.0.0.1:8080/metadata \
        --num-clients 16 \
        --ops-per-client 1000 \
        --detailed-stats
"""

import argparse
import time
import statistics
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable
from tqdm import tqdm
import os
import threading
import queue
import copy
import math
import sys
from collections import defaultdict

from mooncake.store import MooncakeDistributedStore

# Disable memcpy optimization
os.environ["MC_STORE_MEMCPY"] = "0"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('stress_master_rpc_benchmark')


class PerformanceTracker:
    """Tracks and calculates performance metrics for RPC operations."""

    def __init__(self):
        self.operation_latencies: List[float] = []
        self.start_time: float = sys.float_info.max
        self.end_time: float = sys.float_info.min
        self.total_operations: int = 0
        self.failed_operations: int = 0
        self.error_codes: Dict[int, int] = defaultdict(int)

    def record_operation(self, latency_seconds: float):
        self.operation_latencies.append(latency_seconds)
        self.total_operations += 1

    def record_error(self, error_code: int):
        self.error_codes[error_code] += 1
        self.failed_operations += 1
        self.total_operations += 1

    def start_timer(self):
        self.start_time = time.perf_counter()

    def stop_timer(self):
        self.end_time = time.perf_counter()

    def get_wall_time(self) -> float:
        return self.end_time - self.start_time if self.end_time > self.start_time else 0

    def extend(self, other: 'PerformanceTracker'):
        self.operation_latencies.extend(other.operation_latencies)
        self.total_operations += other.total_operations
        self.failed_operations += other.failed_operations
        self.start_time = min(self.start_time, other.start_time)
        self.end_time = max(self.end_time, other.end_time)
        for code, count in other.error_codes.items():
            self.error_codes[code] += count

    def get_statistics(self) -> Dict[str, Any]:
        if not self.operation_latencies:
            return {"error": "No operations recorded"}

        latencies_ms = [lat * 1000 for lat in self.operation_latencies]
        total_time = sum(self.operation_latencies)
        n = len(latencies_ms)

        p50 = statistics.median(latencies_ms) if n >= 2 else latencies_ms[0]
        p90 = statistics.quantiles(latencies_ms, n=10)[8] if n >= 10 else max(latencies_ms)
        p99 = statistics.quantiles(latencies_ms, n=100)[98] if n >= 100 else max(latencies_ms)
        p999 = statistics.quantiles(latencies_ms, n=1000)[998] if n >= 1000 else max(latencies_ms)

        wall_time = self.get_wall_time()
        wall_ops_per_second = self.total_operations / wall_time if wall_time > 0 else 0
        serial_ops_per_second = self.total_operations / total_time if total_time > 0 else 0

        return {
            "total_operations": self.total_operations,
            "succeeded_operations": self.total_operations - self.failed_operations,
            "failed_operations": self.failed_operations,
            "serial_time_seconds": total_time,
            "wall_time_seconds": wall_time,
            "p50_latency_ms": p50,
            "p90_latency_ms": p90,
            "p99_latency_ms": p99,
            "p999_latency_ms": p999,
            "mean_latency_ms": statistics.mean(latencies_ms),
            "min_latency_ms": min(latencies_ms),
            "max_latency_ms": max(latencies_ms),
            "stdev_latency_ms": statistics.stdev(latencies_ms) if n >= 2 else 0,
            "serial_ops_per_second": serial_ops_per_second,
            "wall_ops_per_second": wall_ops_per_second,
            "error_codes": dict(self.error_codes),
        }


@dataclass
class RpcBenchResult:
    """Result of a single RPC benchmark phase."""
    rpc_type: str
    tracker: PerformanceTracker = field(default_factory=PerformanceTracker)


class MasterRpcClient:
    """A single client that connects to master and performs RPC operations."""

    def __init__(self, client_id: int, args):
        self.client_id = client_id
        self.args = args
        self.store = None
        self.buffer_array = None
        self.buffer_ptr = None
        self.setup_latency = 0.0

    def setup(self):
        """Initialize the MooncakeDistributedStore (triggers MountSegment RPC)."""
        self.store = MooncakeDistributedStore()

        protocol = self.args.protocol
        device_name = self.args.device_name
        local_hostname = self.args.local_hostname
        metadata_server = self.args.metadata_server
        global_segment_size = self.args.global_segment_size * 1024 * 1024
        local_buffer_size = self.args.local_buffer_size * 1024 * 1024
        master_server_address = self.args.master_server

        logger.info(f"Client-{self.client_id}: Setting up connection to master {master_server_address}")

        setup_start = time.perf_counter()
        retcode = self.store.setup(
            local_hostname, metadata_server, global_segment_size,
            local_buffer_size, protocol, device_name, master_server_address
        )
        self.setup_latency = time.perf_counter() - setup_start

        if retcode:
            logger.error(f"Client-{self.client_id}: Store setup failed with return code {retcode}")
            raise RuntimeError(f"Store setup failed: {retcode}")

        # Allocate and register a small buffer for put/get operations
        buffer_size = self.args.value_length * self.args.batch_size
        self.buffer_array = np.zeros(buffer_size, dtype=np.uint8)
        self.buffer_ptr = self.buffer_array.ctypes.data

        retcode = self.store.register_buffer(self.buffer_ptr, buffer_size)
        if retcode:
            logger.error(f"Client-{self.client_id}: Buffer registration failed with return code {retcode}")
            raise RuntimeError(f"Buffer registration failed: {retcode}")

        logger.info(f"Client-{self.client_id}: Setup complete in {self.setup_latency * 1000:.2f} ms")

    def teardown(self):
        if self.store:
            self.store.close()
            self.store = None

    def bench_health_check(self, num_ops: int) -> RpcBenchResult:
        """Benchmark health_check (Ping) RPC."""
        result = RpcBenchResult(rpc_type="health_check")
        result.tracker.start_timer()

        for _ in range(num_ops):
            t0 = time.perf_counter()
            rc = self.store.health_check()
            latency = time.perf_counter() - t0
            if rc == 0:
                result.tracker.record_operation(latency)
            else:
                result.tracker.record_error(rc)

        result.tracker.stop_timer()
        return result

    def bench_is_exist(self, num_ops: int) -> RpcBenchResult:
        """Benchmark is_exist (ExistKey) RPC."""
        result = RpcBenchResult(rpc_type="is_exist")
        result.tracker.start_timer()

        for i in range(num_ops):
            key = f"client{self.client_id}_key{i}"
            t0 = time.perf_counter()
            rc = self.store.is_exist(key)
            latency = time.perf_counter() - t0
            # is_exist returns: 1 = exists, 0 = not exists, -1 = error
            if rc >= 0:
                result.tracker.record_operation(latency)
            else:
                result.tracker.record_error(rc)

        result.tracker.stop_timer()
        return result

    def bench_batch_is_exist(self, num_ops: int) -> RpcBenchResult:
        """Benchmark batch_is_exist (BatchExistKey) RPC."""
        result = RpcBenchResult(rpc_type="batch_is_exist")
        batch_size = self.args.batch_size
        num_batches = max(1, num_ops // batch_size)
        result.tracker.start_timer()

        for b in range(num_batches):
            keys = [f"client{self.client_id}_bkey{b * batch_size + j}" for j in range(batch_size)]
            t0 = time.perf_counter()
            results = self.store.batch_is_exist(keys)
            latency = time.perf_counter() - t0

            errors = sum(1 for r in results if r < 0)
            if errors == 0:
                result.tracker.record_operation(latency)
            else:
                result.tracker.record_error(-1)

        result.tracker.stop_timer()
        return result

    def bench_put_get(self, num_ops: int) -> RpcBenchResult:
        """Benchmark put + get cycle (PutStart/PutEnd + GetReplicaList RPCs)."""
        result = RpcBenchResult(rpc_type="put_get")
        value_length = self.args.value_length
        result.tracker.start_timer()

        for i in range(num_ops):
            key = f"client{self.client_id}_pgkey{i}"

            # Fill buffer with pattern
            self.buffer_array[:value_length] = i % 256

            # PUT: triggers PutStart + data transfer + PutEnd on master
            t0 = time.perf_counter()
            rc_put = self.store.put_from(key, self.buffer_ptr, value_length)
            put_latency = time.perf_counter() - t0

            if rc_put != 0:
                result.tracker.record_error(rc_put)
                continue

            # GET: triggers GetReplicaList on master + data transfer
            t1 = time.perf_counter()
            rc_get = self.store.get_into(key, self.buffer_ptr, value_length)
            get_latency = time.perf_counter() - t1

            if rc_get > 0:
                result.tracker.record_operation(put_latency + get_latency)
            else:
                result.tracker.record_error(int(rc_get))

        result.tracker.stop_timer()
        return result

    def bench_put_only(self, num_ops: int) -> RpcBenchResult:
        """Benchmark put-only (PutStart/PutEnd RPCs)."""
        result = RpcBenchResult(rpc_type="put_only")
        value_length = self.args.value_length
        result.tracker.start_timer()

        for i in range(num_ops):
            key = f"client{self.client_id}_pkey{i}"
            self.buffer_array[:value_length] = i % 256

            t0 = time.perf_counter()
            rc = self.store.put_from(key, self.buffer_ptr, value_length)
            latency = time.perf_counter() - t0

            if rc == 0:
                result.tracker.record_operation(latency)
            else:
                result.tracker.record_error(rc)

        result.tracker.stop_timer()
        return result

    def bench_batch_put(self, num_ops: int) -> RpcBenchResult:
        """Benchmark batch put (BatchPutStart/BatchPutEnd RPCs)."""
        result = RpcBenchResult(rpc_type="batch_put")
        value_length = self.args.value_length
        batch_size = self.args.batch_size
        num_batches = max(1, num_ops // batch_size)
        result.tracker.start_timer()

        for b in range(num_batches):
            keys = [f"client{self.client_id}_bpkey{b * batch_size + j}" for j in range(batch_size)]
            buffer_ptrs = []
            sizes = []
            for j in range(batch_size):
                offset = j * value_length
                self.buffer_array[offset:offset + value_length] = (b * batch_size + j) % 256
                buffer_ptrs.append(self.buffer_ptr + offset)
                sizes.append(value_length)

            t0 = time.perf_counter()
            rcs = self.store.batch_put_from(keys, buffer_ptrs, sizes)
            latency = time.perf_counter() - t0

            errors = sum(1 for rc in rcs if rc != 0)
            if errors == 0:
                result.tracker.record_operation(latency)
            else:
                result.tracker.record_error(-1)

        result.tracker.stop_timer()
        return result

    def bench_get_replica_desc(self, num_ops: int) -> RpcBenchResult:
        """Benchmark get_replica_desc (GetReplicaList RPC) on existing keys."""
        result = RpcBenchResult(rpc_type="get_replica_desc")

        # First put some keys to query
        value_length = self.args.value_length
        num_keys = min(num_ops, 100)
        keys = []
        for i in range(num_keys):
            key = f"client{self.client_id}_rdkey{i}"
            self.buffer_array[:value_length] = i % 256
            rc = self.store.put_from(key, self.buffer_ptr, value_length)
            if rc == 0:
                keys.append(key)

        if not keys:
            logger.warning(f"Client-{self.client_id}: No keys available for get_replica_desc benchmark")
            return result

        result.tracker.start_timer()
        for i in range(num_ops):
            key = keys[i % len(keys)]
            t0 = time.perf_counter()
            desc = self.store.get_replica_desc(key)
            latency = time.perf_counter() - t0
            # desc is a list; empty list or valid list both count as success
            result.tracker.record_operation(latency)

        result.tracker.stop_timer()
        return result

    def bench_remove(self, num_ops: int) -> RpcBenchResult:
        """Benchmark remove (Remove RPC)."""
        result = RpcBenchResult(rpc_type="remove")

        # First put keys to be removed
        value_length = self.args.value_length
        keys = []
        for i in range(num_ops):
            key = f"client{self.client_id}_rmkey{i}"
            self.buffer_array[:value_length] = i % 256
            rc = self.store.put_from(key, self.buffer_ptr, value_length)
            if rc == 0:
                keys.append(key)

        if not keys:
            logger.warning(f"Client-{self.client_id}: No keys available for remove benchmark")
            return result

        result.tracker.start_timer()
        for key in keys:
            t0 = time.perf_counter()
            rc = self.store.remove(key)
            latency = time.perf_counter() - t0
            if rc == 0:
                result.tracker.record_operation(latency)
            else:
                result.tracker.record_error(rc)

        result.tracker.stop_timer()
        return result


RPC_BENCHMARKS = {
    "health_check": MasterRpcClient.bench_health_check,
    "is_exist": MasterRpcClient.bench_is_exist,
    "batch_is_exist": MasterRpcClient.bench_batch_is_exist,
    "put_get": MasterRpcClient.bench_put_get,
    "put_only": MasterRpcClient.bench_put_only,
    "batch_put": MasterRpcClient.bench_batch_put,
    "get_replica_desc": MasterRpcClient.bench_get_replica_desc,
    "remove": MasterRpcClient.bench_remove,
}


def worker_thread(client_id, args, results_queue, setup_barrier, start_barrier, end_barrier):
    """Worker thread: setup client, wait for barrier, run benchmarks."""
    thread_name = f"Client-{client_id}"
    try:
        client = MasterRpcClient(client_id, args)
        client.setup()

        # Signal setup complete
        setup_barrier.wait()

        # Wait for all clients to be ready
        start_barrier.wait()

        # Run specified RPC benchmarks
        rpc_types = args.rpc_types.split(",")
        phase_results = {}

        for rpc_type in rpc_types:
            rpc_type = rpc_type.strip()
            if rpc_type not in RPC_BENCHMARKS:
                logger.warning(f"{thread_name}: Unknown RPC type '{rpc_type}', skipping")
                continue

            logger.info(f"{thread_name}: Starting {rpc_type} benchmark ({args.ops_per_client} ops)")
            bench_func = RPC_BENCHMARKS[rpc_type]
            bench_result = bench_func(client, args.ops_per_client)
            phase_results[rpc_type] = bench_result
            logger.info(f"{thread_name}: {rpc_type} complete - "
                        f"{bench_result.tracker.total_operations} ops, "
                        f"{bench_result.tracker.failed_operations} failures")

        results_queue.put({
            "client_id": client_id,
            "setup_latency": client.setup_latency,
            "phase_results": phase_results,
        })

        client.teardown()
        logger.info(f"{thread_name}: Completed and torn down")

    except Exception as e:
        logger.error(f"{thread_name}: Failed with error: {e}", exc_info=True)
        results_queue.put({
            "client_id": client_id,
            "setup_latency": -1,
            "phase_results": {},
            "error": str(e),
        })
    finally:
        try:
            end_barrier.wait()
        except threading.BrokenBarrierError:
            pass


def print_stats(stats: Dict[str, Any], title: str):
    """Print performance statistics."""
    if "error" in stats:
        logger.info(f"No data for {title}: {stats['error']}")
        return

    total_ops = stats["total_operations"]
    succeeded = stats["succeeded_operations"]
    failed = stats["failed_operations"]
    success_rate = (succeeded / total_ops) * 100 if total_ops > 0 else 0

    report = f"\n{'=' * 55}\n"
    report += f"  {title}\n"
    report += f"{'=' * 55}\n"
    report += f"Operations: {total_ops} total, {succeeded} ok ({success_rate:.1f}%), {failed} failed\n"
    report += f"Wall time:  {stats['wall_time_seconds']:.3f} s\n"
    report += f"Latency (ms):\n"
    report += f"  mean={stats['mean_latency_ms']:.3f}  "
    report += f"min={stats['min_latency_ms']:.3f}  "
    report += f"max={stats['max_latency_ms']:.3f}  "
    report += f"stdev={stats['stdev_latency_ms']:.3f}\n"
    report += f"  p50={stats['p50_latency_ms']:.3f}  "
    report += f"p90={stats['p90_latency_ms']:.3f}  "
    report += f"p99={stats['p99_latency_ms']:.3f}  "
    report += f"p999={stats['p999_latency_ms']:.3f}\n"
    report += f"Throughput:\n"
    report += f"  {stats['wall_ops_per_second']:.1f} ops/s (wall)  "
    report += f"{stats['serial_ops_per_second']:.1f} ops/s (serial)\n"

    if stats['error_codes']:
        report += "Error codes:\n"
        for code, count in stats['error_codes'].items():
            report += f"  code {code}: {count}x\n"

    report += f"{'=' * 55}\n"
    logger.info(report)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Mooncake Master RPC Stress Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Connection settings
    parser.add_argument("--protocol", type=str, default="rdma",
                        help="Communication protocol (tcp/rdma)")
    parser.add_argument("--device-name", type=str, default="erdma_0",
                        help="Network device name for RDMA")
    parser.add_argument("--local-hostname", type=str, default="localhost",
                        help="Local hostname")
    parser.add_argument("--metadata-server", type=str,
                        default="http://127.0.0.1:8080/metadata",
                        help="Metadata server address")
    parser.add_argument("--master-server", type=str, default="localhost:50051",
                        help="Master server address")

    # Memory settings
    parser.add_argument("--global-segment-size", type=int, default=4096,
                        help="Global segment size in MB per client")
    parser.add_argument("--local-buffer-size", type=int, default=512,
                        help="Local buffer size in MB per client")
    parser.add_argument("--value-length", type=int, default=1024,
                        help="Size of each value in bytes for put/get operations")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Batch size for batch operations")

    # Benchmark parameters
    parser.add_argument("--num-clients", type=int, default=4,
                        help="Number of concurrent client instances")
    parser.add_argument("--ops-per-client", type=int, default=500,
                        help="Number of operations per client per RPC type")
    parser.add_argument("--rpc-types", type=str,
                        default="health_check,is_exist,batch_is_exist,put_only",
                        help="Comma-separated list of RPC types to benchmark. "
                             "Available: health_check, is_exist, batch_is_exist, "
                             "put_get, put_only, batch_put, get_replica_desc, remove")

    # Output settings
    parser.add_argument("--detailed-stats", action="store_true",
                        help="Print detailed per-client statistics")

    return parser.parse_args()


def main():
    args = parse_arguments()

    logger.info("=" * 60)
    logger.info("  Mooncake Master RPC Stress Benchmark")
    logger.info("=" * 60)
    logger.info(f"Master server:   {args.master_server}")
    logger.info(f"Protocol:        {args.protocol}")
    logger.info(f"Num clients:     {args.num_clients}")
    logger.info(f"Ops per client:  {args.ops_per_client}")
    logger.info(f"RPC types:       {args.rpc_types}")
    logger.info(f"Value length:    {args.value_length} bytes")
    logger.info(f"Batch size:      {args.batch_size}")
    logger.info("=" * 60)

    num_clients = args.num_clients
    results_queue = queue.Queue()

    # +1 for main thread on each barrier
    setup_barrier = threading.Barrier(num_clients + 1)
    start_barrier = threading.Barrier(num_clients + 1)
    end_barrier = threading.Barrier(num_clients + 1)

    # ---- Phase 1: Concurrent client registration ----
    logger.info(f"\n>>> Phase 1: Registering {num_clients} clients concurrently...")
    registration_start = time.perf_counter()

    threads = []
    for i in range(num_clients):
        t = threading.Thread(
            target=worker_thread,
            args=(i, args, results_queue, setup_barrier, start_barrier, end_barrier),
            name=f"Client-{i}",
            daemon=True,
        )
        threads.append(t)
        t.start()

    # Wait for all clients to finish setup (registration)
    setup_barrier.wait()
    registration_time = time.perf_counter() - registration_start
    logger.info(f">>> All {num_clients} clients registered in {registration_time * 1000:.2f} ms")

    # ---- Phase 2: Concurrent RPC benchmarks ----
    logger.info(f"\n>>> Phase 2: Starting concurrent RPC benchmarks...")
    bench_start = time.perf_counter()
    start_barrier.wait()

    # Wait for all to finish
    end_barrier.wait()
    total_bench_time = time.perf_counter() - bench_start
    logger.info(f">>> All benchmarks completed in {total_bench_time:.2f} s")

    # Wait for threads to exit
    for t in threads:
        t.join(timeout=10)

    # ---- Collect and report results ----
    all_results = []
    while not results_queue.empty():
        all_results.append(results_queue.get())

    # Report registration latencies
    setup_latencies = [r["setup_latency"] for r in all_results if r["setup_latency"] > 0]
    if setup_latencies:
        report = "\n" + "=" * 55 + "\n"
        report += "  CLIENT REGISTRATION (MountSegment RPC)\n"
        report += "=" * 55 + "\n"
        report += f"Clients registered: {len(setup_latencies)}/{num_clients}\n"
        report += f"Total wall time:    {registration_time * 1000:.2f} ms\n"
        latencies_ms = [l * 1000 for l in setup_latencies]
        report += f"Latency (ms):\n"
        report += f"  mean={statistics.mean(latencies_ms):.2f}  "
        report += f"min={min(latencies_ms):.2f}  "
        report += f"max={max(latencies_ms):.2f}\n"
        if len(latencies_ms) >= 2:
            report += f"  stdev={statistics.stdev(latencies_ms):.2f}\n"
        report += f"Registration throughput: {len(setup_latencies) / registration_time:.1f} clients/s\n"
        report += "=" * 55 + "\n"
        logger.info(report)

    # Aggregate per-RPC-type across all clients
    rpc_types = args.rpc_types.split(",")
    for rpc_type in rpc_types:
        rpc_type = rpc_type.strip()

        # Per-client detailed stats
        if args.detailed_stats:
            for r in all_results:
                if rpc_type in r.get("phase_results", {}):
                    bench_result = r["phase_results"][rpc_type]
                    stats = bench_result.tracker.get_statistics()
                    print_stats(stats, f"Client-{r['client_id']} / {rpc_type}")

        # Combined stats
        combined = PerformanceTracker()
        for r in all_results:
            if rpc_type in r.get("phase_results", {}):
                combined.extend(r["phase_results"][rpc_type].tracker)

        if combined.total_operations > 0:
            stats = combined.get_statistics()
            print_stats(stats, f"COMBINED {rpc_type.upper()} ({num_clients} clients)")

    # Overall summary
    total_ops = sum(
        r["phase_results"][rt].tracker.total_operations
        for r in all_results
        for rt in r.get("phase_results", {})
    )
    total_failures = sum(
        r["phase_results"][rt].tracker.failed_operations
        for r in all_results
        for rt in r.get("phase_results", {})
    )
    errors = [r.get("error") for r in all_results if "error" in r]

    summary = "\n" + "=" * 55 + "\n"
    summary += "  OVERALL SUMMARY\n"
    summary += "=" * 55 + "\n"
    summary += f"Clients:          {num_clients}\n"
    summary += f"Total RPC ops:    {total_ops}\n"
    summary += f"Total failures:   {total_failures}\n"
    summary += f"Total wall time:  {total_bench_time:.2f} s\n"
    summary += f"Aggregate QPS:    {total_ops / total_bench_time:.1f} ops/s\n"
    if errors:
        summary += f"Client errors:    {len(errors)}\n"
        for e in errors:
            summary += f"  - {e}\n"
    summary += "=" * 55 + "\n"
    logger.info(summary)


if __name__ == '__main__':
    main()
