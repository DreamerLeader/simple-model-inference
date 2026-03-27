"""
Stress benchmark for Mooncake Master RPC service.

Phase 1: Create and maintain exactly NUM_CLIENTS persistent clients (MountSegment RPC).
Phase 2: All clients concurrently and continuously call is_exist(key) to stress
         the maximum gRPC communication throughput the master can sustain.

Usage:
    python stress_master_rpc_benchmark.py \
        --master-server localhost:50051 \
        --metadata-server http://127.0.0.1:8080/metadata

    # Custom client count and worker threads
    python stress_master_rpc_benchmark.py \
        --master-server localhost:50051 \
        --metadata-server http://127.0.0.1:8080/metadata \
        --num-clients 800 \
        --num-workers 800 \
        --query-key some_key
"""

import argparse
import time
import statistics
import logging
import numpy as np
from typing import List
import os
import threading
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


class MasterRpcClient:
    """A single client that connects to master and stays alive."""

    def __init__(self, client_id: int, args):
        self.client_id = client_id
        self.args = args
        self.store = None
        self.buffer_array = None
        self.buffer_ptr = None
        self.setup_latency = 0.0
        self._lock = threading.Lock()

    def setup(self):
        """Initialize or re-initialize the MooncakeDistributedStore (MountSegment RPC)."""
        self.store = MooncakeDistributedStore()

        protocol = self.args.protocol
        device_name = self.args.device_name
        local_hostname = self.args.local_hostname
        metadata_server = self.args.metadata_server
        global_segment_size = self.args.global_segment_size * 1024 * 1024
        local_buffer_size = self.args.local_buffer_size * 1024 * 1024
        master_server_address = self.args.master_server

        setup_start = time.perf_counter()
        retcode = self.store.setup(
            local_hostname, metadata_server, global_segment_size,
            local_buffer_size, protocol, device_name, master_server_address
        )
        self.setup_latency = time.perf_counter() - setup_start

        if retcode:
            raise RuntimeError(f"Store setup failed: retcode={retcode}")

        # Allocate and register a small buffer
        buffer_size = self.args.value_length
        self.buffer_array = np.zeros(buffer_size, dtype=np.uint8)
        self.buffer_ptr = self.buffer_array.ctypes.data

        retcode = self.store.register_buffer(self.buffer_ptr, buffer_size)
        if retcode:
            raise RuntimeError(f"Buffer registration failed: retcode={retcode}")

    def query_is_exist(self, key: str):
        """
        Call is_exist(key) [IsExist gRPC].
        Returns (latency_s, result) where result is 1/0/-1.
        """
        t0 = time.perf_counter()
        result = self.store.is_exist(key)
        latency = time.perf_counter() - t0
        return latency, result


# ---------------------------------------------------------------------------
# Shared statistics (lock-protected)
# ---------------------------------------------------------------------------

class Stats:
    def __init__(self):
        self._lock = threading.Lock()
        self.success = 0
        self.errors = 0
        self.latencies: List[float] = []
        # Window counters for rolling QPS
        self._window_start = time.perf_counter()
        self._window_count = 0

    def record(self, latency: float):
        with self._lock:
            self.success += 1
            self._window_count += 1
            self.latencies.append(latency)

    def record_error(self):
        with self._lock:
            self.errors += 1

    def snapshot(self):
        with self._lock:
            now = time.perf_counter()
            elapsed = now - self._window_start
            window_qps = self._window_count / elapsed if elapsed > 0 else 0.0
            # Reset window
            self._window_start = now
            self._window_count = 0
            return dict(
                success=self.success,
                errors=self.errors,
                window_qps=window_qps,
                latencies=list(self.latencies),
            )


# ---------------------------------------------------------------------------
# Worker thread: one per client, runs exit+reconnect in a tight loop
# ---------------------------------------------------------------------------

def stress_worker(client: MasterRpcClient, key: str, stats: Stats, stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            latency, _ = client.query_is_exist(key)
            stats.record(latency)
        except Exception as e:
            stats.record_error()
            logger.debug(f"Client-{client.client_id} error: {e}")
            time.sleep(0.1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Mooncake Master RPC Stress Benchmark — Exit/Reconnect Throughput",
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
                        help="Size of registered buffer in bytes per client")

    # Benchmark settings
    parser.add_argument("--num-clients", type=int, default=800,
                        help="Number of persistent clients to create (Phase 1)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Concurrent worker threads for Phase 2 stress "
                             "(0 = same as --num-clients, i.e. one worker per client)")
    parser.add_argument("--query-key", type=str, default="benchmark_probe_key",
                        help="Key to query with is_exist() in Phase 2")
    parser.add_argument("--report-interval", type=int, default=5,
                        help="Statistics reporting interval in seconds")
    parser.add_argument("--duration", type=int, default=0,
                        help="How long to run Phase 2 in seconds (0 = run forever)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()
    if args.num_workers == 0:
        args.num_workers = args.num_clients

    logger.info("=" * 60)
    logger.info("  Mooncake Master — Exit/Reconnect gRPC Stress Benchmark")
    logger.info("=" * 60)
    logger.info(f"Master server:       {args.master_server}")
    logger.info(f"Metadata server:     {args.metadata_server}")
    logger.info(f"Protocol:            {args.protocol}")
    logger.info(f"Persistent clients:  {args.num_clients}")
    logger.info(f"Worker threads:      {args.num_workers}")
    logger.info(f"Query key:           {args.query_key}")
    logger.info(f"Global segment:      {args.global_segment_size} MB")
    logger.info(f"Local buffer:        {args.local_buffer_size} MB")
    logger.info(f"Duration:            {'unlimited' if args.duration == 0 else str(args.duration) + 's'}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Phase 1: Create and keep alive NUM_CLIENTS clients
    # ------------------------------------------------------------------
    logger.info(f"\n[Phase 1] Registering {args.num_clients} persistent clients ...")
    live_clients: List[MasterRpcClient] = []
    phase1_latencies: List[float] = []
    phase1_start = time.perf_counter()

    for cid in range(args.num_clients):
        client = MasterRpcClient(cid, args)
        try:
            client.setup()
        except Exception as e:
            logger.error(f"Client-{cid}: Registration FAILED — {e}")
            logger.error(f"Only {len(live_clients)} / {args.num_clients} clients registered. Aborting.")
            sys.exit(1)

        live_clients.append(client)
        phase1_latencies.append(client.setup_latency)

        if (cid + 1) % 50 == 0 or (cid + 1) == args.num_clients:
            elapsed = time.perf_counter() - phase1_start
            ms = [l * 1000 for l in phase1_latencies[-50:]]
            logger.info(
                f"  Registered {cid + 1}/{args.num_clients} clients  |  "
                f"Elapsed: {elapsed:.1f}s  |  "
                f"Last batch avg: {statistics.mean(ms):.2f} ms/client"
            )

    phase1_elapsed = time.perf_counter() - phase1_start
    ms_all = [l * 1000 for l in phase1_latencies]
    logger.info("")
    logger.info(f"[Phase 1 Complete] {len(live_clients)} clients alive in {phase1_elapsed:.2f}s")
    logger.info(f"  Setup latency — mean={statistics.mean(ms_all):.2f}ms  "
                f"min={min(ms_all):.2f}ms  max={max(ms_all):.2f}ms")
    if len(ms_all) >= 2:
        logger.info(f"  stdev={statistics.stdev(ms_all):.2f}ms")

    # ------------------------------------------------------------------
    # Phase 2: Stress test — concurrent exit + reconnect loop
    # ------------------------------------------------------------------
    logger.info(f"\n[Phase 2] Starting is_exist stress test with {args.num_workers} workers ...")
    logger.info(f"  Each worker continuously calls: is_exist('{args.query_key}') [IsExist gRPC]")
    logger.info("  Press Ctrl+C to stop.\n")

    stats = Stats()
    stop_event = threading.Event()

    # Distribute workers across clients (round-robin if workers > clients)
    threads: List[threading.Thread] = []
    for i in range(args.num_workers):
        client = live_clients[i % len(live_clients)]
        t = threading.Thread(
            target=stress_worker,
            args=(client, args.query_key, stats, stop_event),
            daemon=True,
            name=f"worker-{i}"
        )
        threads.append(t)

    phase2_start = time.perf_counter()
    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(args.report_interval)
            snap = stats.snapshot()
            elapsed = time.perf_counter() - phase2_start
            total_ops = snap['success']
            overall_qps = total_ops / elapsed if elapsed > 0 else 0.0

            lat_ms = [l * 1000 for l in snap['latencies']] if snap['latencies'] else []

            logger.info("-" * 60)
            logger.info(f"  Elapsed: {elapsed:.1f}s  |  Total ops: {total_ops}  |  Errors: {snap['errors']}")
            logger.info(f"  Throughput (last {args.report_interval}s): {snap['window_qps']:.1f} ops/s  "
                        f"|  Overall: {overall_qps:.1f} ops/s")
            if lat_ms:
                sorted_ms = sorted(lat_ms)
                p99 = sorted_ms[int(len(sorted_ms) * 0.99)] if len(sorted_ms) >= 100 else sorted_ms[-1]
                logger.info(f"  is_exist latency (ms): mean={statistics.mean(lat_ms):.2f}  "
                            f"min={min(lat_ms):.2f}  max={max(lat_ms):.2f}  p99={p99:.2f}")

            if args.duration > 0 and elapsed >= args.duration:
                logger.info(f"\n[Phase 2] Duration {args.duration}s reached — stopping workers.")
                break

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user (Ctrl+C).")

    stop_event.set()
    logger.info("Waiting for workers to finish ...")
    for t in threads:
        t.join(timeout=5)

    # Final summary
    elapsed = time.perf_counter() - phase2_start
    snap = stats.snapshot()
    total_ops = snap['success']
    overall_qps = total_ops / elapsed if elapsed > 0 else 0.0
    lat_ms = [l * 1000 for l in snap['latencies']] if snap['latencies'] else []

    logger.info("")
    logger.info("=" * 60)
    logger.info("  Phase 2 Final Summary")
    logger.info("=" * 60)
    logger.info(f"  Duration:           {elapsed:.2f}s")
    logger.info(f"  Persistent clients: {len(live_clients)}")
    logger.info(f"  Worker threads:     {args.num_workers}")
    logger.info(f"  Query key:          {args.query_key}")
    logger.info(f"  Total ops:          {total_ops}")
    logger.info(f"  Total errors:       {snap['errors']}")
    logger.info(f"  Overall QPS:        {overall_qps:.1f} ops/s")
    if lat_ms:
        sorted_ms = sorted(lat_ms)
        p99 = sorted_ms[int(len(sorted_ms) * 0.99)] if len(sorted_ms) >= 100 else sorted_ms[-1]
        p999 = sorted_ms[int(len(sorted_ms) * 0.999)] if len(sorted_ms) >= 1000 else sorted_ms[-1]
        logger.info(f"  is_exist lat (ms):  mean={statistics.mean(lat_ms):.2f}  "
                    f"min={min(lat_ms):.2f}  max={max(lat_ms):.2f}  "
                    f"p99={p99:.2f}  p99.9={p999:.2f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
