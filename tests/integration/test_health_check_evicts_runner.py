"""Layer 2: Health check failure transitions WARM runner to COLD."""

import asyncio

import pytest

from helios.config import PoolConfig, RunnerConfig
from helios.fsm import RunnerLifecycleState
from helios.pool import HeliosPool


@pytest.mark.asyncio
async def test_unhealthy_warm_runner_transitions_to_cold() -> None:
    """A WARM runner that fails is_healthy() is transitioned to COLD.

    Uses failure_rate=1.0 so is_healthy() always returns False.
    State is set to WARM manually (bypassing load, which would also fail
    with failure_rate=1.0). Memory is deducted to simulate a loaded model.
    """
    pool = HeliosPool(
        config=PoolConfig(
            total_memory_gb=8.0,
            max_concurrent_loads=2,
            request_timeout_s=5.0,
            prewarm_interval_s=60.0,
            health_check_interval_s=0.1,  # fast checks for test speed
        )
    )
    pool.register(
        RunnerConfig(
            model_id="sick",
            memory_gb=2.0,
            load_time_s=0.01,
            infer_time_s=0.001,
            failure_rate=1.0,  # always fails health check
        )
    )
    pool.start()

    # Manually set to WARM (bypassing load which would fail with failure_rate=1.0).
    pool._runner_states["sick"] = RunnerLifecycleState.WARM
    pool._available_memory_gb -= 2.0

    try:
        # Wait for at least one health check cycle.
        await asyncio.sleep(0.3)

        assert pool._runner_states["sick"] is RunnerLifecycleState.COLD
        assert pool._available_memory_gb == pytest.approx(8.0)
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_healthy_warm_runner_stays_warm() -> None:
    """A WARM runner that passes is_healthy() remains WARM."""
    pool = HeliosPool(
        config=PoolConfig(
            total_memory_gb=8.0,
            max_concurrent_loads=2,
            request_timeout_s=5.0,
            prewarm_interval_s=60.0,
            health_check_interval_s=0.1,
        )
    )
    pool.register(
        RunnerConfig(
            model_id="healthy",
            memory_gb=2.0,
            load_time_s=0.01,
            infer_time_s=0.001,
            failure_rate=0.0,  # always passes health check
        )
    )
    pool.start()

    pool._runner_states["healthy"] = RunnerLifecycleState.WARM
    pool._available_memory_gb -= 2.0

    try:
        await asyncio.sleep(0.3)

        assert pool._runner_states["healthy"] is RunnerLifecycleState.WARM
        assert pool._available_memory_gb == pytest.approx(6.0)
    finally:
        await pool.shutdown()
