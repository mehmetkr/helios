#!/usr/bin/env python3
"""Helios benchmark harness.

Compares LRU vs cost-based eviction under configurable load profiles.

Usage:
    python benchmark.py --profile steady --policy lru
    python benchmark.py --profile spiky --policy cost_based
    python benchmark.py --profile diurnal --policy cost_based
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import time
import warnings
from dataclasses import dataclass, field

from helios.app import SIMULATION_MODELS
from helios.config import InferenceRequest, PoolConfig
from helios.exceptions import HeliosError
from helios.policies.cost_based import CostBasedEvictionPolicy
from helios.policies.lru import LRUEvictionPolicy
from helios.pool import HeliosPool
from helios.router import RequestRouter

# Suppress statsmodels warnings (divide by zero in log, convergence).
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

DURATION_S = 15.0
TOTAL_MEMORY_GB = 40.0  # half of default 80GB -- creates eviction pressure
MODEL_IDS = [cfg.model_id for cfg in SIMULATION_MODELS]


@dataclass
class BenchmarkResult:
    """Collected metrics from a benchmark run."""

    profile: str
    policy: str
    duration_s: float = 0.0
    latencies: list[float] = field(default_factory=list)
    warm_hits: int = 0
    cold_starts: int = 0
    errors: int = 0
    evictions: int = 0


def _requests_per_second(profile: str, elapsed: float) -> float:
    """Return the target request rate for the given profile at elapsed time."""
    if profile == "steady":
        return 2.0
    if profile == "spiky":
        # Baseline 1 req/s, burst to 5 req/s for 10s every 60s.
        cycle = elapsed % 60.0
        return 5.0 if cycle < 10.0 else 1.0
    if profile == "diurnal":
        # Sinusoidal: 0.5 to 3.0 req/s over a 60s compressed day cycle.
        return 1.75 + 1.25 * math.sin(2 * math.pi * elapsed / 60.0)
    msg = f"Unknown profile: {profile}"
    raise ValueError(msg)


async def _run_benchmark(
    profile: str,
    policy: str,
) -> BenchmarkResult:
    """Run the benchmark and collect results."""
    eviction_policy = (
        CostBasedEvictionPolicy(total_memory_gb=TOTAL_MEMORY_GB)
        if policy == "cost_based"
        else LRUEvictionPolicy()
    )
    pool = HeliosPool(
        config=PoolConfig(
            total_memory_gb=TOTAL_MEMORY_GB,
            max_concurrent_loads=4,
            request_timeout_s=30.0,
            prewarm_interval_s=1.0,
            prewarm_threshold=0.3,
            ewma_alpha=0.3,
            health_check_interval_s=10.0,
            max_retries=2,
        ),
        eviction_policy=eviction_policy,
    )
    for cfg in SIMULATION_MODELS:
        pool.register(cfg, rng=random.Random(42))
    pool.start()
    router = RequestRouter(pool)

    result = BenchmarkResult(profile=profile, policy=policy)
    rng = random.Random(123)
    start = time.monotonic()
    pending: set[asyncio.Task[None]] = set()

    async def _send_request(model_id: str) -> None:
        req = InferenceRequest(model_id=model_id, payload="benchmark")
        req_start = time.monotonic()
        try:
            res = await router.route(req)
            latency = time.monotonic() - req_start
            result.latencies.append(latency)
            if res.cache_status == "warm":
                result.warm_hits += 1
            else:
                result.cold_starts += 1
        except HeliosError:
            result.errors += 1

    try:
        while True:
            elapsed = time.monotonic() - start
            if elapsed >= DURATION_S:
                break

            rate = _requests_per_second(profile, elapsed)
            delay = 1.0 / rate if rate > 0 else 1.0
            await asyncio.sleep(delay)

            model_id = rng.choice(MODEL_IDS)
            task = asyncio.create_task(_send_request(model_id))
            pending.add(task)
            task.add_done_callback(pending.discard)

        # Wait for all in-flight requests to complete.
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

        result.duration_s = time.monotonic() - start

        # Count evictions from Prometheus counter.
        from prometheus_client import REGISTRY

        for metric in REGISTRY.collect():
            if metric.name == "helios_evictions":
                for sample in metric.samples:
                    if sample.name == "helios_evictions_total":
                        result.evictions += int(sample.value)
    finally:
        await pool.shutdown()

    return result


def _percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile of a sorted list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def _print_result(result: BenchmarkResult) -> None:
    """Print benchmark results in a formatted table."""
    total_requests = result.warm_hits + result.cold_starts + result.errors
    warm_pct = (result.warm_hits / total_requests * 100) if total_requests else 0
    cold_pct = (result.cold_starts / total_requests * 100) if total_requests else 0

    print(f"\n{'=' * 60}")
    print(f"  Profile: {result.profile:<12} Policy: {result.policy}")
    print(
        f"  Duration: {result.duration_s:.1f}s    "
        f"Requests: {total_requests}    Errors: {result.errors}"
    )
    print(f"{'=' * 60}")
    print()
    print("  Latency (seconds):")
    print(f"    p50:  {_percentile(result.latencies, 50):.3f}")
    print(f"    p95:  {_percentile(result.latencies, 95):.3f}")
    print(f"    p99:  {_percentile(result.latencies, 99):.3f}")
    print()
    print("  Cache:")
    print(f"    Warm hits:   {result.warm_hits:>5} ({warm_pct:.1f}%)")
    print(f"    Cold starts: {result.cold_starts:>5} ({cold_pct:.1f}%)")
    print()
    print(f"  Evictions: {result.evictions}")
    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Helios benchmark harness")
    parser.add_argument(
        "--profile",
        choices=["steady", "spiky", "diurnal"],
        default="steady",
        help="Load profile (default: steady)",
    )
    parser.add_argument(
        "--policy",
        choices=["lru", "cost_based"],
        default="lru",
        help="Eviction policy (default: lru)",
    )
    args = parser.parse_args()

    result = asyncio.run(_run_benchmark(args.profile, args.policy))
    _print_result(result)


if __name__ == "__main__":
    main()
