"""Layer 4: Timeout does not cancel underlying load task.

Tests asyncio.shield behavior: a caller's timeout raises RequestTimeoutError
but the underlying load task continues. Subsequent callers find the model WARM.
"""

import asyncio

import pytest

from helios.config import InferenceRequest, PoolConfig, RunnerConfig
from helios.exceptions import RequestTimeoutError
from helios.fsm import RunnerLifecycleState
from helios.pool import HeliosPool
from helios.router import RequestRouter


@pytest.mark.asyncio
async def test_timed_out_caller_does_not_cancel_underlying_load_task() -> None:
    """First caller times out. Load continues via asyncio.shield.
    Second caller finds the model WARM and returns immediately.
    """
    pool = HeliosPool(
        config=PoolConfig(
            total_memory_gb=8.0,
            max_concurrent_loads=2,
            request_timeout_s=0.2,  # short timeout -- load will exceed this
            prewarm_interval_s=60.0,
            health_check_interval_s=60.0,
        )
    )
    pool.register(
        RunnerConfig(
            model_id="slow",
            memory_gb=2.0,
            load_time_s=1.0,  # 1s load > 0.2s timeout
            infer_time_s=0.001,
        )
    )
    pool.start()
    router = RequestRouter(pool)

    try:
        # First request: should time out while model is loading.
        with pytest.raises(RequestTimeoutError):
            await router.route(InferenceRequest(model_id="slow", payload="timeout"))

        # Wait for the shielded load to complete.
        await asyncio.sleep(1.5)

        # Model should now be WARM (load completed despite caller timeout).
        assert pool._runner_states["slow"] is RunnerLifecycleState.WARM

        # Second request: model is already WARM, should return immediately.
        result = await asyncio.wait_for(
            router.route(InferenceRequest(model_id="slow", payload="fast")),
            timeout=0.5,
        )
        assert result.model_id == "slow"
        assert result.cache_status == "warm"
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_multiple_timeouts_single_load_completes() -> None:
    """Multiple callers time out, but only one load is in progress.
    When it completes, the model is WARM for subsequent requests.
    """
    pool = HeliosPool(
        config=PoolConfig(
            total_memory_gb=8.0,
            max_concurrent_loads=2,
            request_timeout_s=0.1,  # very short timeout
            prewarm_interval_s=60.0,
            health_check_interval_s=60.0,
        )
    )
    pool.register(
        RunnerConfig(
            model_id="slow",
            memory_gb=2.0,
            load_time_s=0.5,  # 0.5s load > 0.1s timeout
            infer_time_s=0.001,
        )
    )
    pool.start()
    router = RequestRouter(pool)

    try:
        # Fire 5 sequential requests -- most will time out, but the last
        # may succeed if the shielded load completes during the sequence
        # (0.5s load, 5 x 0.1s timeout = load finishes mid-sequence).
        timeout_count = 0
        for _ in range(5):
            try:
                await router.route(InferenceRequest(model_id="slow", payload="x"))
            except RequestTimeoutError:
                timeout_count += 1
        assert timeout_count >= 3, f"Expected most requests to time out, got {timeout_count}"

        # Wait for the shielded load to definitely finish.
        await asyncio.sleep(1.0)
        assert pool._runner_states["slow"] is RunnerLifecycleState.WARM

        # Next request succeeds immediately (model already WARM).
        result = await router.route(InferenceRequest(model_id="slow", payload="ok"))
        assert result.cache_status == "warm"
    finally:
        await pool.shutdown()
