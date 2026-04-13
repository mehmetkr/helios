"""Layer 2: Memory lock prevents two concurrent loads from jointly exceeding budget."""

import asyncio
import contextlib

import pytest

from helios.config import InferenceRequest
from helios.exceptions import HeliosError
from helios.pool import HeliosPool
from helios.router import RequestRouter


@pytest.mark.asyncio
async def test_concurrent_loads_respect_budget(concurrent_pool: HeliosPool) -> None:
    """5 concurrent loads on an 8GB pool with 2GB models must not exceed budget.

    Pool holds max 4 models (8GB / 2GB). The 5th triggers eviction.
    _available_memory_gb must never go negative. Some requests may raise
    RunnerStateError if their model is evicted between ensure_loaded and
    dispatch -- this is legitimate under contention and not a budget violation.
    """
    router = RequestRouter(concurrent_pool)
    budget_violated = False

    async def budget_monitor() -> None:
        nonlocal budget_violated
        while True:
            if concurrent_pool._available_memory_gb < -0.001:
                budget_violated = True
            await asyncio.sleep(0)

    async def route_ignoring_errors(model_id: str) -> None:
        with contextlib.suppress(HeliosError):
            await router.route(InferenceRequest(model_id=model_id, payload="test"))

    monitor = asyncio.create_task(budget_monitor())
    try:
        await asyncio.gather(*[route_ignoring_errors(str(i)) for i in range(5)])
    finally:
        monitor.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await monitor

    assert not budget_violated, "Memory budget was exceeded during concurrent loads"
    assert concurrent_pool._available_memory_gb >= 0
