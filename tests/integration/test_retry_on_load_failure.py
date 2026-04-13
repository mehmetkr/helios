"""Layer 2: Retry on load failure.

_RunnerLoadAttemptError is retried; RunnerLoadError raised after max retries.
"""

import pytest

from helios.config import InferenceRequest, PoolConfig, RunnerConfig
from helios.exceptions import RunnerLoadError
from helios.pool import HeliosPool
from helios.router import RequestRouter


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_failure() -> None:
    """With failure_rate < 1.0, retry loop eventually succeeds."""
    pool = HeliosPool(
        config=PoolConfig(
            total_memory_gb=8.0,
            max_concurrent_loads=2,
            request_timeout_s=5.0,
            prewarm_interval_s=60.0,
            health_check_interval_s=60.0,
            max_retries=5,
        )
    )
    # Use a seeded RNG that produces: fail, fail, success (for this seed).
    import random

    pool.register(
        RunnerConfig(
            model_id="flaky",
            memory_gb=2.0,
            load_time_s=0.01,
            infer_time_s=0.001,
            failure_rate=0.5,
        ),
        rng=random.Random(0),
    )
    pool.start()
    router = RequestRouter(pool)
    try:
        result = await router.route(InferenceRequest(model_id="flaky", payload="test"))
        assert result.model_id == "flaky"
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_raises_runner_load_error_after_max_retries() -> None:
    """With failure_rate=1.0, all attempts fail -> RunnerLoadError."""
    pool = HeliosPool(
        config=PoolConfig(
            total_memory_gb=8.0,
            max_concurrent_loads=2,
            request_timeout_s=5.0,
            prewarm_interval_s=60.0,
            health_check_interval_s=60.0,
            max_retries=2,
        )
    )
    pool.register(
        RunnerConfig(
            model_id="broken",
            memory_gb=2.0,
            load_time_s=0.01,
            infer_time_s=0.001,
            failure_rate=1.0,
        )
    )
    pool.start()
    router = RequestRouter(pool)
    try:
        with pytest.raises(RunnerLoadError, match="3 attempts"):
            await router.route(InferenceRequest(model_id="broken", payload="test"))
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_multiple_simulate_load_calls_on_failure() -> None:
    """Verify that _simulate_load is called multiple times before final error."""
    pool = HeliosPool(
        config=PoolConfig(
            total_memory_gb=8.0,
            max_concurrent_loads=2,
            request_timeout_s=5.0,
            prewarm_interval_s=60.0,
            health_check_interval_s=60.0,
            max_retries=2,
        )
    )
    pool.register(
        RunnerConfig(
            model_id="tracked",
            memory_gb=2.0,
            load_time_s=0.01,
            infer_time_s=0.001,
            failure_rate=1.0,
        )
    )
    pool.start()
    router = RequestRouter(pool)

    load_count = 0
    original = pool._simulate_load

    async def counting_load(model_id: str) -> None:
        nonlocal load_count
        load_count += 1
        await original(model_id)

    pool._simulate_load = counting_load  # type: ignore[assignment]

    try:
        with pytest.raises(RunnerLoadError):
            await router.route(InferenceRequest(model_id="tracked", payload="test"))
        assert load_count == 3, f"Expected 3 load attempts (1 + 2 retries), got {load_count}"
    finally:
        await pool.shutdown()
