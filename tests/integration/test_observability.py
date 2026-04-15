"""Layer 2: Observability validation -- Prometheus metrics emit correctly.

Verifies that a defined workload produces expected metric increments.
Uses before/after deltas to handle cross-test counter accumulation.
"""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

from helios.config import InferenceRequest
from helios.pool import HeliosPool
from helios.router import RequestRouter


def _sample(name: str, labels: dict[str, str] | None = None) -> float:
    """Read a single sample value from the Prometheus registry."""
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


@pytest.mark.asyncio
async def test_cold_start_increments_metric(fast_pool: HeliosPool, router: RequestRouter) -> None:
    """Loading a COLD model increments helios_cold_starts_total."""
    before = _sample("helios_cold_starts_total", {"model_id": "0"})
    await router.route(InferenceRequest(model_id="0", payload="test"))
    after = _sample("helios_cold_starts_total", {"model_id": "0"})
    assert after - before == 1.0


@pytest.mark.asyncio
async def test_memory_gauge_reflects_loaded_model(
    fast_pool: HeliosPool, router: RequestRouter
) -> None:
    """After loading a model, POOL_MEMORY_USED_GB is positive."""
    await router.route(InferenceRequest(model_id="0", payload="test"))
    used = _sample("helios_pool_memory_used_gb")
    assert used > 0


@pytest.mark.asyncio
async def test_queue_depth_returns_to_zero(fast_pool: HeliosPool, router: RequestRouter) -> None:
    """After a request completes, queue depth returns to 0."""
    await router.route(InferenceRequest(model_id="0", payload="test"))
    depth = _sample("helios_queue_depth")
    assert depth == 0.0


@pytest.mark.asyncio
async def test_load_duration_recorded(fast_pool: HeliosPool, router: RequestRouter) -> None:
    """After a cold load, LOAD_DURATION_SECONDS count is incremented."""
    before = _sample(
        "helios_load_duration_seconds_count",
        {"model_id": "0", "outcome": "success"},
    )
    await router.route(InferenceRequest(model_id="0", payload="test"))
    after = _sample(
        "helios_load_duration_seconds_count",
        {"model_id": "0", "outcome": "success"},
    )
    assert after - before == 1.0
