"""Shared test fixtures for Helios integration, property, and scenario tests.

Three canonical pool shapes:
- fast_pool: minimal latencies for speed (load_time=0.01s)
- concurrent_pool: varied load times for concurrency pressure (0.5-1.5s)
- failure_pool: 50% failure rate with distinct RNG seeds per runner

All fixtures register 10 models with IDs "0" through "9".
Test fixtures and the 20-model app.py catalog are intentionally independent.
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator

import pytest_asyncio
from hypothesis import strategies as st

from helios.config import InferenceRequest, PoolConfig, RunnerConfig
from helios.pool import HeliosPool
from helios.router import RequestRouter


@pytest_asyncio.fixture
async def fast_pool() -> AsyncIterator[HeliosPool]:
    """Fast pool: 8GB budget, 10 x 2GB models, near-instant load/infer."""
    pool = HeliosPool(
        config=PoolConfig(
            total_memory_gb=8.0,
            max_concurrent_loads=2,
            request_timeout_s=5.0,
            prewarm_interval_s=1.0,
            health_check_interval_s=1.0,
        )
    )
    for i in range(10):
        pool.register(
            RunnerConfig(
                model_id=str(i),
                memory_gb=2.0,
                load_time_s=0.01,
                infer_time_s=0.001,
                failure_rate=0.0,
            )
        )
    pool.start()
    yield pool
    await pool.shutdown()


@pytest_asyncio.fixture
async def concurrent_pool() -> AsyncIterator[HeliosPool]:
    """Concurrent pool: varied load times for concurrency pressure."""
    pool = HeliosPool(
        config=PoolConfig(
            total_memory_gb=8.0,
            max_concurrent_loads=4,
            request_timeout_s=10.0,
            prewarm_interval_s=1.0,
            health_check_interval_s=1.0,
        )
    )
    load_times = [0.5, 0.75, 1.0, 1.25, 1.5, 0.5, 0.75, 1.0, 1.25, 1.5]
    for i in range(10):
        pool.register(
            RunnerConfig(
                model_id=str(i),
                memory_gb=2.0,
                load_time_s=load_times[i],
                infer_time_s=0.05,
                failure_rate=0.0,
            )
        )
    pool.start()
    yield pool
    await pool.shutdown()


@pytest_asyncio.fixture
async def failure_pool() -> AsyncIterator[HeliosPool]:
    """Failure pool: 50% failure rate with distinct RNG seeds per runner."""
    pool = HeliosPool(
        config=PoolConfig(
            total_memory_gb=8.0,
            max_concurrent_loads=2,
            request_timeout_s=5.0,
            prewarm_interval_s=1.0,
            health_check_interval_s=1.0,
            max_retries=2,
        )
    )
    for i in range(10):
        pool.register(
            RunnerConfig(
                model_id=str(i),
                memory_gb=2.0,
                load_time_s=0.1,
                infer_time_s=0.05,
                failure_rate=0.5,
            ),
            rng=random.Random(42 + i),
        )
    pool.start()
    yield pool
    await pool.shutdown()


@pytest_asyncio.fixture
async def router(fast_pool: HeliosPool) -> RequestRouter:
    """Router wrapping the fast_pool fixture."""
    return RequestRouter(fast_pool)


@st.composite
def model_request_strategy(
    draw: st.DrawFn,
) -> list[InferenceRequest]:
    """Hypothesis strategy returning a list of InferenceRequests.

    Guarantees all 10 model IDs ("0" through "9") appear at least once,
    forcing eviction pressure on every Hypothesis example. Additional
    random requests are appended and the full list is shuffled.
    Use as @given(model_request_strategy()) -- NOT @given(st.lists(...)).
    """
    base = [InferenceRequest(model_id=str(i), payload="test") for i in range(10)]
    extra = draw(
        st.lists(
            st.builds(
                InferenceRequest,
                model_id=st.sampled_from([str(i) for i in range(10)]),
                payload=st.just("test"),
            ),
            max_size=20,
        )
    )
    return draw(st.permutations(base + extra))
