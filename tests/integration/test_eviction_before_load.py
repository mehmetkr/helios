"""Layer 2: Eviction fires before loading when budget would be exceeded."""

import pytest

from helios.config import InferenceRequest
from helios.fsm import RunnerLifecycleState
from helios.pool import HeliosPool
from helios.router import RequestRouter


@pytest.mark.asyncio
async def test_eviction_frees_memory_for_new_model(fast_pool: HeliosPool) -> None:
    """8GB budget, 2GB models -> max 4. Loading a 5th evicts the LRU model."""
    router = RequestRouter(fast_pool)

    # Load 4 models to fill the 8GB budget.
    for i in range(4):
        await router.route(InferenceRequest(model_id=str(i), payload="fill"))
    assert fast_pool._available_memory_gb == pytest.approx(0.0)

    # Load a 5th -- must evict one of the first 4.
    await router.route(InferenceRequest(model_id="4", payload="trigger_eviction"))

    assert fast_pool._runner_states["4"] is RunnerLifecycleState.WARM
    # At least one of the original 4 should now be COLD.
    cold_count = sum(
        1 for i in range(4) if fast_pool._runner_states[str(i)] is RunnerLifecycleState.COLD
    )
    assert cold_count >= 1, "No eviction occurred -- budget should have been exceeded"


@pytest.mark.asyncio
async def test_lru_evicts_oldest_model(fast_pool: HeliosPool) -> None:
    """LRU policy should evict the model with the oldest last_request_at."""
    router = RequestRouter(fast_pool)

    # Load models 0-3 in order. Model 0 is requested first (oldest).
    for i in range(4):
        await router.route(InferenceRequest(model_id=str(i), payload="fill"))

    # Load model 4 -- should evict model 0 (oldest last_request_at).
    await router.route(InferenceRequest(model_id="4", payload="trigger_eviction"))

    assert fast_pool._runner_states["0"] is RunnerLifecycleState.COLD
    assert fast_pool._runner_states["4"] is RunnerLifecycleState.WARM
