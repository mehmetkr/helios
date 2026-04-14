"""Layer 4: Budget exhaustion scenario tests.

Tests that structurally impossible memory requests raise permanent
MemoryExhaustedError and that the pool recovers cleanly.
"""

import pytest

from helios.config import InferenceRequest, PoolConfig, RunnerConfig
from helios.exceptions import MemoryExhaustedError, RunnerLoadError
from helios.pool import HeliosPool
from helios.router import RequestRouter


@pytest.mark.asyncio
async def test_budget_exhaustion_raises_memory_exhausted_error() -> None:
    """A model requiring more memory than total budget raises permanent
    MemoryExhaustedError, not a generic error or timeout.
    """
    pool = HeliosPool(
        config=PoolConfig(
            total_memory_gb=4.0,
            max_concurrent_loads=2,
            request_timeout_s=5.0,
            prewarm_interval_s=60.0,
            health_check_interval_s=60.0,
        )
    )
    pool.register(
        RunnerConfig(
            model_id="giant",
            memory_gb=5.0,  # exceeds 4GB budget
            load_time_s=0.01,
            infer_time_s=0.001,
        )
    )
    pool.start()
    router = RequestRouter(pool)

    try:
        with pytest.raises(MemoryExhaustedError, match="structurally impossible") as exc_info:
            await router.route(InferenceRequest(model_id="giant", payload="test"))
        assert exc_info.value.permanent is True
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_recovers_after_budget_exhaustion() -> None:
    """After a permanent MemoryExhaustedError, the pool can still serve
    requests for models that fit within the budget.
    """
    pool = HeliosPool(
        config=PoolConfig(
            total_memory_gb=4.0,
            max_concurrent_loads=2,
            request_timeout_s=5.0,
            prewarm_interval_s=60.0,
            health_check_interval_s=60.0,
        )
    )
    pool.register(
        RunnerConfig(
            model_id="giant",
            memory_gb=5.0,
            load_time_s=0.01,
            infer_time_s=0.001,
        )
    )
    pool.register(
        RunnerConfig(
            model_id="small",
            memory_gb=2.0,
            load_time_s=0.01,
            infer_time_s=0.001,
        )
    )
    pool.start()
    router = RequestRouter(pool)

    try:
        # Giant model fails permanently.
        with pytest.raises(MemoryExhaustedError):
            await router.route(InferenceRequest(model_id="giant", payload="fail"))

        # Small model should still work.
        result = await router.route(InferenceRequest(model_id="small", payload="ok"))
        assert result.model_id == "small"
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_all_retries_exhausted_raises_runner_load_error() -> None:
    """With failure_rate=1.0, all load attempts fail and RunnerLoadError
    is raised after max_retries + 1 attempts.
    """
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
            await router.route(InferenceRequest(model_id="broken", payload="fail"))
    finally:
        await pool.shutdown()
