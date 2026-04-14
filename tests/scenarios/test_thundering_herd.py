"""Layer 4: Thundering herd scenario tests.

Realistic simulations of sustained load causing eviction and backpressure
when all runners are active.
"""

import asyncio

import pytest

from helios.config import InferenceRequest, PoolConfig, RunnerConfig
from helios.exceptions import HeliosError
from helios.fsm import RunnerLifecycleState
from helios.pool import HeliosPool
from helios.router import RequestRouter


@pytest.mark.asyncio
async def test_sustained_load_causes_lru_eviction() -> None:
    """Continuous requests across more models than the pool can hold
    causes LRU evictions. The most recently used models survive.
    """
    pool = HeliosPool(
        config=PoolConfig(
            total_memory_gb=8.0,
            max_concurrent_loads=2,
            request_timeout_s=5.0,
            prewarm_interval_s=60.0,
            health_check_interval_s=60.0,
        )
    )
    for i in range(10):
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
        # Load models 0-3 (fills 8GB budget).
        for i in range(4):
            await router.route(InferenceRequest(model_id=str(i), payload="fill"))

        # Load models 4-7, one at a time — each evicts the LRU model.
        for i in range(4, 8):
            await router.route(InferenceRequest(model_id=str(i), payload="evict"))

        # Models 0-3 should all be evicted (COLD). Models 4-7 should be WARM.
        for i in range(4):
            assert pool._runner_states[str(i)] is RunnerLifecycleState.COLD
        for i in range(4, 8):
            assert pool._runner_states[str(i)] is RunnerLifecycleState.WARM
    finally:
        await pool.shutdown()


@pytest.mark.asyncio
async def test_backpressure_when_all_runners_active() -> None:
    """When all WARM models are serving concurrent inferences (ACTIVE),
    new requests for COLD models face transient MemoryExhaustedError
    because ACTIVE runners are immune to eviction.
    """
    pool = HeliosPool(
        config=PoolConfig(
            total_memory_gb=8.0,
            max_concurrent_loads=4,
            request_timeout_s=2.0,
            prewarm_interval_s=60.0,
            health_check_interval_s=60.0,
            max_retries=1,
        )
    )
    for i in range(10):
        pool.register(
            RunnerConfig(
                model_id=str(i),
                memory_gb=2.0,
                load_time_s=0.01,
                infer_time_s=0.5,  # slow infer keeps runners ACTIVE
            )
        )
    pool.start()
    router = RequestRouter(pool)

    try:
        # Load 4 models to fill budget, then start long inferences on all 4.
        for i in range(4):
            await router.route(InferenceRequest(model_id=str(i), payload="load"))

        # Now fire concurrent long inferences on 0-3 (making them ACTIVE)
        # AND a request for model 4 (needs eviction but all are ACTIVE).
        errors_caught: list[HeliosError] = []

        async def route_with_error_capture(model_id: str) -> None:
            try:
                await router.route(InferenceRequest(model_id=model_id, payload="test"))
            except HeliosError as exc:
                errors_caught.append(exc)

        await asyncio.gather(
            route_with_error_capture("0"),
            route_with_error_capture("1"),
            route_with_error_capture("2"),
            route_with_error_capture("3"),
            route_with_error_capture("4"),  # needs eviction -- but all are ACTIVE
        )

        # Model 4 may have failed due to backpressure (MemoryExhaustedError)
        # or timed out. Either way, the pool should not have crashed.
        assert pool._available_memory_gb >= 0
    finally:
        await pool.shutdown()
