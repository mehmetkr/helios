"""Layer 2: ensure_loaded return value -- True on warm hit, False on cold load."""

import pytest

from helios.config import InferenceRequest
from helios.fsm import RunnerLifecycleState
from helios.pool import HeliosPool
from helios.router import RequestRouter


@pytest.mark.asyncio
async def test_returns_false_on_cold_load(fast_pool: HeliosPool) -> None:
    """First call for a COLD model triggers a load -- returns False."""
    assert fast_pool._runner_states["0"] is RunnerLifecycleState.COLD
    was_warm = await fast_pool.ensure_loaded("0")
    assert was_warm is False


@pytest.mark.asyncio
async def test_returns_true_on_warm_hit(fast_pool: HeliosPool) -> None:
    """Second call for an already-WARM model -- returns True."""
    await fast_pool.ensure_loaded("0")
    assert fast_pool._runner_states["0"] is RunnerLifecycleState.WARM
    was_warm = await fast_pool.ensure_loaded("0")
    assert was_warm is True


@pytest.mark.asyncio
async def test_router_sets_cold_cache_status(fast_pool: HeliosPool) -> None:
    """Router sets cache_status='cold' when ensure_loaded returns False."""
    router = RequestRouter(fast_pool)
    result = await router.route(InferenceRequest(model_id="0", payload="test"))
    assert result.cache_status == "cold"


@pytest.mark.asyncio
async def test_router_sets_warm_cache_status(fast_pool: HeliosPool) -> None:
    """Router sets cache_status='warm' when ensure_loaded returns True."""
    router = RequestRouter(fast_pool)
    # First request: cold start.
    await router.route(InferenceRequest(model_id="0", payload="first"))
    # Second request: warm hit.
    result = await router.route(InferenceRequest(model_id="0", payload="second"))
    assert result.cache_status == "warm"
