"""Layer 3: Hypothesis property-based invariant tests.

Proves correctness of a stateful concurrent system under adversarial inputs.
Five invariants verified:

1. Pool memory never exceeds total_memory_gb (concurrent budget monitor)
2. Memory accounting is consistent after idle (no leaks)
3. Every dropped request raises a HeliosError subclass (never bare Exception)
4. No concurrent duplicate loads for the same model (idempotency)
5. select_for_eviction always returns a model_id from its candidate set
"""

from __future__ import annotations

import asyncio
import contextlib
import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from helios.config import (
    InferenceRequest,
    PoolConfig,
    RunnerConfig,
    RunnerMetrics,
)
from helios.exceptions import HeliosError
from helios.fsm import RunnerLifecycleState
from helios.policies.cost_based import CostBasedEvictionPolicy
from helios.policies.lru import LRUEvictionPolicy
from helios.pool import HeliosPool
from helios.router import RequestRouter
from tests.conftest import model_request_strategy


def _make_pool() -> HeliosPool:
    """Create a test pool: 8GB budget, 10 x 2GB models, fast load/infer."""
    pool = HeliosPool(
        config=PoolConfig(
            total_memory_gb=8.0,
            max_concurrent_loads=2,
            request_timeout_s=5.0,
            prewarm_interval_s=1.0,
            health_check_interval_s=1.0,
        )
    )
    for i in range(10):
        pool.register(
            RunnerConfig(
                model_id=str(i),
                memory_gb=2.0,
                load_time_s=0.01,
                infer_time_s=0.001,
                failure_rate=0.0,
            )
        )
    return pool


# ---------------------------------------------------------------------------
# Invariant 1: Pool memory never exceeds total_memory_gb
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(model_request_strategy())
@settings(max_examples=500, deadline=None)
async def test_pool_never_exceeds_memory_budget(
    requests: list[InferenceRequest],
) -> None:
    """Budget invariant: _available_memory_gb never goes negative and the
    sum of WARM/ACTIVE model memory never exceeds total_memory_gb, even
    under concurrent adversarial request sequences.
    """
    pool = _make_pool()
    pool.start()
    router = RequestRouter(pool)
    budget_violated = False

    async def budget_monitor() -> None:
        nonlocal budget_violated
        while True:
            if pool._available_memory_gb < -0.001:
                budget_violated = True
            used = sum(
                pool._configs[mid].memory_gb
                for mid, state in pool._runner_states.items()
                if state in (RunnerLifecycleState.WARM, RunnerLifecycleState.ACTIVE)
            )
            if used > 8.0 + 0.001:
                budget_violated = True
            await asyncio.sleep(0)

    async def route_ignoring_errors(req: InferenceRequest) -> None:
        with contextlib.suppress(HeliosError):
            await asyncio.wait_for(router.route(req), timeout=30.0)

    monitor = asyncio.create_task(budget_monitor())
    try:
        await asyncio.gather(*[route_ignoring_errors(req) for req in requests])
    finally:
        monitor.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await monitor
        await pool.shutdown()
    assert not budget_violated, "Budget exceeded during concurrent execution"


# ---------------------------------------------------------------------------
# Invariant 2: Memory accounting is consistent (no leaks)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(model_request_strategy())
@settings(max_examples=200, deadline=None)
async def test_memory_accounting_consistent_after_idle(
    requests: list[InferenceRequest],
) -> None:
    """After all requests complete, available + WARM memory == total.
    Catches memory leaks from failed rollbacks or double-reservations.
    """
    pool = _make_pool()
    pool.start()
    router = RequestRouter(pool)

    async def route_ignoring_errors(req: InferenceRequest) -> None:
        with contextlib.suppress(HeliosError):
            await asyncio.wait_for(router.route(req), timeout=30.0)

    await asyncio.gather(*[route_ignoring_errors(req) for req in requests])
    await pool.shutdown()

    # After shutdown, all runners are COLD and full budget is restored.
    assert pool._available_memory_gb == pytest.approx(pool.config.total_memory_gb)


# ---------------------------------------------------------------------------
# Invariant 3: Every dropped request raises a HeliosError subclass
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(model_request_strategy())
@settings(max_examples=200, deadline=None)
async def test_all_exceptions_are_typed(
    requests: list[InferenceRequest],
) -> None:
    """Every exception escaping router.route() must be a HeliosError subclass.
    No bare Exception, no _RunnerLoadAttemptError, no asyncio.TimeoutError.
    """
    pool = _make_pool()
    pool.start()
    router = RequestRouter(pool)
    untyped_exceptions: list[Exception] = []

    async def route_checking_exceptions(req: InferenceRequest) -> None:
        try:
            await asyncio.wait_for(router.route(req), timeout=30.0)
        except HeliosError:
            pass  # expected typed error
        except Exception as exc:
            untyped_exceptions.append(exc)

    try:
        await asyncio.gather(*[route_checking_exceptions(req) for req in requests])
    finally:
        await pool.shutdown()

    assert not untyped_exceptions, (
        f"Untyped exceptions escaped route(): "
        f"{[(type(e).__name__, str(e)) for e in untyped_exceptions]}"
    )


# ---------------------------------------------------------------------------
# Invariant 4: No concurrent duplicate loads (idempotency)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@given(model_request_strategy())
@settings(max_examples=200, deadline=None)
async def test_no_concurrent_duplicate_loads(
    requests: list[InferenceRequest],
) -> None:
    """While a model is loading, no second _simulate_load fires for it.
    This is the idempotency guarantee: N concurrent requests for the same
    COLD model trigger exactly ONE load.
    """
    pool = _make_pool()
    pool.start()
    router = RequestRouter(pool)

    currently_loading: set[str] = set()
    violation = False
    original_load = pool._simulate_load

    async def guarded_load(model_id: str) -> None:
        nonlocal violation
        if model_id in currently_loading:
            violation = True
        currently_loading.add(model_id)
        try:
            await original_load(model_id)
        finally:
            currently_loading.discard(model_id)

    pool._simulate_load = guarded_load  # type: ignore[assignment]

    async def route_ignoring_errors(req: InferenceRequest) -> None:
        with contextlib.suppress(HeliosError):
            await asyncio.wait_for(router.route(req), timeout=30.0)

    try:
        await asyncio.gather(*[route_ignoring_errors(req) for req in requests])
    finally:
        await pool.shutdown()

    assert not violation, "Concurrent duplicate load detected -- idempotency violated"


# ---------------------------------------------------------------------------
# Invariant 5: select_for_eviction returns from candidate set
# ---------------------------------------------------------------------------


runner_config_strategy = st.builds(
    RunnerConfig,
    model_id=st.text(min_size=1, max_size=10),
    memory_gb=st.floats(min_value=0.1, max_value=10.0),
    load_time_s=st.floats(min_value=0.01, max_value=20.0),
    failure_rate=st.just(0.0),
    infer_time_s=st.floats(min_value=0.01, max_value=1.0),
)

runner_metrics_strategy = st.builds(
    RunnerMetrics,
    model_id=st.text(min_size=1, max_size=10),
    last_request_at=st.floats(min_value=0.0, max_value=1e9),
    request_rate_per_min=st.floats(min_value=0.0, max_value=1000.0),
    predicted_demand=st.floats(min_value=0.0, max_value=1.0),
)


@given(st.data())
@settings(max_examples=500)
def test_lru_eviction_returns_from_candidate_set(data: st.DataObject) -> None:
    """LRU policy must return a model_id that exists in its input."""
    n = data.draw(st.integers(min_value=1, max_value=20))
    model_ids = [f"m{i}" for i in range(n)]
    configs = {
        mid: RunnerConfig(
            model_id=mid,
            memory_gb=data.draw(st.floats(min_value=0.1, max_value=10.0)),
            load_time_s=data.draw(st.floats(min_value=0.01, max_value=20.0)),
        )
        for mid in model_ids
    }
    now = time.monotonic()
    metrics = {
        mid: RunnerMetrics(
            model_id=mid,
            last_request_at=now - data.draw(st.floats(min_value=0.0, max_value=1000.0)),
        )
        for mid in model_ids
    }
    victim = LRUEvictionPolicy().select_for_eviction(configs, metrics)
    assert victim in configs, f"LRU returned {victim!r} which is not in candidates"


@given(st.data())
@settings(max_examples=500)
def test_cost_based_eviction_returns_from_candidate_set(
    data: st.DataObject,
) -> None:
    """Cost-based policy must return a model_id that exists in its input."""
    n = data.draw(st.integers(min_value=1, max_value=20))
    model_ids = [f"m{i}" for i in range(n)]
    configs = {
        mid: RunnerConfig(
            model_id=mid,
            memory_gb=data.draw(st.floats(min_value=0.1, max_value=10.0)),
            load_time_s=data.draw(st.floats(min_value=0.01, max_value=20.0)),
        )
        for mid in model_ids
    }
    now = time.monotonic()
    metrics = {
        mid: RunnerMetrics(
            model_id=mid,
            last_request_at=now - data.draw(st.floats(min_value=0.0, max_value=1000.0)),
            request_rate_per_min=data.draw(st.floats(min_value=0.0, max_value=100.0)),
            predicted_demand=data.draw(st.floats(min_value=0.0, max_value=1.0)),
        )
        for mid in model_ids
    }
    victim = CostBasedEvictionPolicy(total_memory_gb=80.0).select_for_eviction(configs, metrics)
    assert victim in configs, f"CostBased returned {victim!r} which is not in candidates"
