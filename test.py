"""
Stress benchmark for Mooncake Master RPC service — max client registration test.

Continuously creates clients in a while loop, each client calls setup() to
register with the master (MountSegment RPC). Clients are kept alive (never
torn down) to find the maximum number of concurrent clients the master can
support.

Usage:
    python stress_master_rpc_benchmark.py \
        --master-server localhost:50051 \
        --metadata-server http://127.0.0.1:8080/metadata

    # With custom segment sizes
    python stress_master_rpc_benchmark.py \
        --master-server localhost:50051 \
        --metadata-server http://127.0.0.1:8080/metadata \
        --global-segment-size 512 \
        --local-buffer-size 64
"""

import argparse
import time
import statistics
import logging
import numpy as np
from typing import List, Dict, Any
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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Mooncake Master RPC Stress Benchmark — Max Client Registration",
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

    return parser.parse_args()


def main():
    args = parse_arguments()

    logger.info("=" * 60)
    logger.info("  Mooncake Master — Max Client Registration Stress Test")
    logger.info("=" * 60)
    logger.info(f"Master server:       {args.master_server}")
    logger.info(f"Metadata server:     {args.metadata_server}")
    logger.info(f"Protocol:            {args.protocol}")
    logger.info(f"Global segment size: {args.global_segment_size} MB")
    logger.info(f"Local buffer size:   {args.local_buffer_size} MB")
    logger.info("=" * 60)

    # All live clients are kept in this list — never released
    live_clients: List[MasterRpcClient] = []
    setup_latencies: List[float] = []  # in seconds
    total_start = time.perf_counter()
    client_id = 0

    try:
        while True:
            client = MasterRpcClient(client_id, args)
            try:
                client.setup()
            except Exception as e:
                elapsed = time.perf_counter() - total_start
                logger.error(f"Client-{client_id}: Registration FAILED — {e}")
                logger.info("")
                logger.info("=" * 60)
                logger.info("  REACHED MAX CLIENT LIMIT")
                logger.info("=" * 60)
                logger.info(f"Last successful client ID:  {client_id - 1}")
                logger.info(f"Total alive clients:        {len(live_clients)}")
                logger.info(f"Total elapsed time:         {elapsed:.2f} s")
                if setup_latencies:
                    ms = [l * 1000 for l in setup_latencies]
                    logger.info(f"Registration latency (ms):")
                    logger.info(f"  mean={statistics.mean(ms):.2f}  "
                                f"min={min(ms):.2f}  max={max(ms):.2f}")
                    if len(ms) >= 2:
                        logger.info(f"  stdev={statistics.stdev(ms):.2f}")
                    logger.info(f"Avg throughput: {len(live_clients) / elapsed:.1f} clients/s")
                logger.info("=" * 60)
                break

            # Keep client alive
            live_clients.append(client)
            setup_latencies.append(client.setup_latency)
            client_id += 1

            # Periodic progress report every 10 clients
            if client_id % 10 == 0:
                elapsed = time.perf_counter() - total_start
                recent_ms = [l * 1000 for l in setup_latencies[-10:]]
                logger.info(
                    f"[Progress] Alive clients: {len(live_clients)}  |  "
                    f"Elapsed: {elapsed:.1f}s  |  "
                    f"Last 10 avg latency: {statistics.mean(recent_ms):.2f} ms"
                )

    except KeyboardInterrupt:
        elapsed = time.perf_counter() - total_start
        logger.info("")
        logger.info("=" * 60)
        logger.info("  INTERRUPTED BY USER (Ctrl+C)")
        logger.info("=" * 60)
        logger.info(f"Total alive clients:  {len(live_clients)}")
        logger.info(f"Total elapsed time:   {elapsed:.2f} s")
        if setup_latencies:
            ms = [l * 1000 for l in setup_latencies]
            logger.info(f"Registration latency (ms):")
            logger.info(f"  mean={statistics.mean(ms):.2f}  "
                        f"min={min(ms):.2f}  max={max(ms):.2f}")
        logger.info("=" * 60)

    # Keep the process alive so all clients stay registered
    if live_clients:
        logger.info(f"\n>>> All {len(live_clients)} clients remain connected. "
                     f"Press Ctrl+C again to exit.")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Exiting.")


if __name__ == '__main__':
    main()
