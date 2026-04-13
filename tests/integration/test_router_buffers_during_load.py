"""Layer 2: Router buffers request during LOADING and returns result when done.

Also tests idempotency: N concurrent cold requests trigger exactly ONE load.
"""

import asyncio

import pytest

from helios.config import InferenceRequest
from helios.fsm import RunnerLifecycleState
from helios.pool import HeliosPool
from helios.router import RequestRouter


@pytest.mark.asyncio
async def test_cold_request_loads_and_returns(
    fast_pool: HeliosPool, router: RequestRouter
) -> None:
    """A request for a COLD model triggers a load and returns a result."""
    assert fast_pool._runner_states["0"] is RunnerLifecycleState.COLD
    result = await router.route(InferenceRequest(model_id="0", payload="test"))
    assert result.model_id == "0"
    assert fast_pool._runner_states["0"] in (
        RunnerLifecycleState.WARM,
        RunnerLifecycleState.ACTIVE,
    )


@pytest.mark.asyncio
async def test_warm_request_returns_immediately(fast_pool: HeliosPool) -> None:
    """A request for an already-WARM model skips loading (fast path 1)."""
    router = RequestRouter(fast_pool)
    # Load model first.
    await router.route(InferenceRequest(model_id="0", payload="first"))
    assert fast_pool._runner_states["0"] is RunnerLifecycleState.WARM

    # Second request should return near-instantly (no load).
    result = await router.route(InferenceRequest(model_id="0", payload="second"))
    assert result.model_id == "0"


@pytest.mark.asyncio
async def test_concurrent_cold_requests_trigger_one_load(
    fast_pool: HeliosPool,
) -> None:
    """N concurrent requests for the same COLD model trigger exactly ONE load.

    This is the idempotency property -- the core design guarantee.
    """
    router = RequestRouter(fast_pool)
    load_count = 0
    original = fast_pool._simulate_load

    async def counting_load(model_id: str) -> None:
        nonlocal load_count
        load_count += 1
        await original(model_id)

    fast_pool._simulate_load = counting_load  # type: ignore[assignment]

    await asyncio.gather(
        *[router.route(InferenceRequest(model_id="0", payload="x")) for _ in range(20)]
    )
    assert load_count == 1, f"Expected 1 load, got {load_count}"
