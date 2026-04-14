"""Layer 4: Traffic spike scenario test.

Tests that the pre-warm loop detects rising demand and proactively loads
models before explicit requests arrive.
"""

import asyncio

import pytest

from helios.config import InferenceRequest, PoolConfig, RunnerConfig
from helios.fsm import RunnerLifecycleState
from helios.pool import HeliosPool
from helios.router import RequestRouter


@pytest.mark.asyncio
async def test_sudden_spike_triggers_prewarming() -> None:
    """Sustained traffic for a model builds demand history. After the model
    is evicted, the pre-warm loop detects high predicted demand and
    proactively reloads it without an explicit request.

    Uses short prewarm_interval_s (0.05s) so HoltPredictor accumulates
    MIN_HISTORY (10) observations in ~0.5s.
    """
    pool = HeliosPool(
        config=PoolConfig(
            total_memory_gb=6.0,  # tight budget: 3 x 2GB models
            max_concurrent_loads=2,
            request_timeout_s=5.0,
            prewarm_interval_s=0.05,  # fast ticks for test speed
            prewarm_threshold=0.3,
            health_check_interval_s=60.0,  # disable health checks
        )
    )
    for i in range(6):
        pool.register(
            RunnerConfig(
                model_id=str(i),
                memory_gb=2.0,
                load_time_s=0.01,
                infer_time_s=0.001,
            )
        )
    pool.start()
    router = RequestRouter(pool)

    try:
        # Phase 1: Generate sustained traffic for model "0" to build demand
        # history. Each prewarm tick (0.05s) drains _request_counter and
        # feeds the predictor. We need ~15 ticks for HoltPredictor to
        # accumulate MIN_HISTORY (10) and start forecasting.
        for _ in range(20):
            await router.route(InferenceRequest(model_id="0", payload="spike"))
            await asyncio.sleep(0.05)  # pace with prewarm ticks

        # Model "0" should be WARM with high demand history.
        assert pool._runner_states["0"] is RunnerLifecycleState.WARM

        # Phase 2: Evict model "0" by loading enough other models.
        await router.route(InferenceRequest(model_id="1", payload="fill"))
        await router.route(InferenceRequest(model_id="2", payload="fill"))
        await router.route(InferenceRequest(model_id="3", payload="evict"))

        # Model "0" should now be COLD (evicted by LRU).
        assert pool._runner_states["0"] is RunnerLifecycleState.COLD

        # Phase 3: Wait for pre-warm loop to detect high predicted demand
        # for model "0" and proactively reload it. Allow up to 2s.
        for _ in range(40):
            await asyncio.sleep(0.05)
            if pool._runner_states["0"] is RunnerLifecycleState.WARM:
                break

        assert pool._runner_states["0"] is RunnerLifecycleState.WARM, (
            "Pre-warm did not trigger for high-demand model after eviction"
        )
    finally:
        await pool.shutdown()
