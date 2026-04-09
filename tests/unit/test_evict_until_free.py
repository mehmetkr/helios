"""Unit tests for HeliosPool._evict_until_free().

Layer 1: synchronous, no asyncio required. _evict_until_free is a plain def.
Tests the three termination cases:
1. Structural impossibility (required > total budget)
2. No WARM candidates (all ACTIVE or LOADING)
3. Multi-eviction loop (requires multiple evictions before budget satisfied)
"""

import pytest

from helios.config import PoolConfig, RunnerConfig
from helios.exceptions import MemoryExhaustedError, RunnerStateError
from helios.fsm import RunnerLifecycleState
from helios.policies.lru import LRUEvictionPolicy
from helios.pool import HeliosPool


def _make_pool(total_memory_gb: float = 10.0) -> HeliosPool:
    """Create a pool with a given budget. Models are registered manually."""
    return HeliosPool(
        config=PoolConfig(total_memory_gb=total_memory_gb),
        eviction_policy=LRUEvictionPolicy(),
    )


def _register_warm(
    pool: HeliosPool, model_id: str, memory_gb: float, last_request_at: float = 0.0
) -> None:
    """Register a model and set it to WARM with reserved memory."""
    config = RunnerConfig(model_id=model_id, memory_gb=memory_gb, load_time_s=1.0)
    pool.register(config)
    pool._runner_states[model_id] = RunnerLifecycleState.WARM
    pool._metrics[model_id].last_request_at = last_request_at
    pool._available_memory_gb -= memory_gb


class TestStructuralImpossibility:
    def test_raises_permanent_when_required_exceeds_total(self) -> None:
        pool = _make_pool(total_memory_gb=10.0)
        with pytest.raises(MemoryExhaustedError, match="structurally impossible") as exc_info:
            pool._evict_until_free(15.0)
        assert exc_info.value.permanent is True

    def test_raises_even_with_all_memory_free(self) -> None:
        pool = _make_pool(total_memory_gb=10.0)
        with pytest.raises(MemoryExhaustedError) as exc_info:
            pool._evict_until_free(10.1)
        assert exc_info.value.permanent is True


class TestNoWarmCandidates:
    def test_raises_transient_when_all_cold(self) -> None:
        """All runners COLD, memory reserved (simulating in-progress loads).
        Need more memory but nothing WARM to evict."""
        pool = _make_pool(total_memory_gb=10.0)
        config = RunnerConfig(model_id="m1", memory_gb=4.0, load_time_s=1.0)
        pool.register(config)
        # State is COLD but memory is reserved (simulates a loading model)
        pool._available_memory_gb -= 4.0  # 6GB free
        with pytest.raises(MemoryExhaustedError, match="no WARM runners") as exc_info:
            pool._evict_until_free(8.0)  # needs 8, has 6, no WARM to evict
        assert exc_info.value.permanent is False

    def test_raises_transient_when_all_active(self) -> None:
        pool = _make_pool(total_memory_gb=10.0)
        config = RunnerConfig(model_id="m1", memory_gb=4.0, load_time_s=1.0)
        pool.register(config)
        pool._runner_states["m1"] = RunnerLifecycleState.ACTIVE
        pool._available_memory_gb -= 4.0  # 6GB free
        with pytest.raises(MemoryExhaustedError, match="no WARM runners") as exc_info:
            pool._evict_until_free(8.0)  # needs 8, has 6, no WARM to evict
        assert exc_info.value.permanent is False

    def test_raises_transient_when_all_loading(self) -> None:
        pool = _make_pool(total_memory_gb=10.0)
        config = RunnerConfig(model_id="m1", memory_gb=4.0, load_time_s=1.0)
        pool.register(config)
        pool._runner_states["m1"] = RunnerLifecycleState.DOWNLOADING
        pool._available_memory_gb -= 4.0
        with pytest.raises(MemoryExhaustedError, match="no WARM runners") as exc_info:
            pool._evict_until_free(8.0)
        assert exc_info.value.permanent is False


class TestMultiEvictionLoop:
    def test_evicts_multiple_runners_to_satisfy_budget(self) -> None:
        """4 WARM runners x 2GB = 8GB used. Budget 10GB, available 2GB.
        Need 8GB → must evict 3 runners (freeing 6GB → available = 8GB)."""
        pool = _make_pool(total_memory_gb=10.0)
        for i in range(4):
            _register_warm(pool, f"m{i}", memory_gb=2.0, last_request_at=float(i))
        assert pool._available_memory_gb == pytest.approx(2.0)

        pool._evict_until_free(8.0)

        assert pool._available_memory_gb >= 8.0
        warm_count = sum(1 for s in pool._runner_states.values() if s is RunnerLifecycleState.WARM)
        assert warm_count == 1  # only 1 survivor

    def test_evicts_lru_order(self) -> None:
        """LRU evicts oldest first (lowest last_request_at)."""
        pool = _make_pool(total_memory_gb=10.0)
        _register_warm(pool, "old", memory_gb=3.0, last_request_at=1.0)
        _register_warm(pool, "mid", memory_gb=3.0, last_request_at=5.0)
        _register_warm(pool, "new", memory_gb=3.0, last_request_at=10.0)
        # 9GB used, 1GB free. Need 4GB → evict 1 runner (3GB freed → 4GB).
        pool._evict_until_free(4.0)

        assert pool._runner_states["old"] is RunnerLifecycleState.COLD
        assert pool._runner_states["mid"] is RunnerLifecycleState.WARM
        assert pool._runner_states["new"] is RunnerLifecycleState.WARM

    def test_single_eviction_sufficient(self) -> None:
        pool = _make_pool(total_memory_gb=10.0)
        _register_warm(pool, "m1", memory_gb=4.0, last_request_at=1.0)
        # 4GB used, 6GB free. Need 8GB → evict 1 runner (4GB freed → 10GB).
        pool._evict_until_free(8.0)
        assert pool._runner_states["m1"] is RunnerLifecycleState.COLD
        assert pool._available_memory_gb == pytest.approx(10.0)

    def test_no_eviction_needed_when_budget_sufficient(self) -> None:
        pool = _make_pool(total_memory_gb=10.0)
        _register_warm(pool, "m1", memory_gb=2.0, last_request_at=1.0)
        # 2GB used, 8GB free. Need 4GB → no eviction needed.
        pool._evict_until_free(4.0)
        assert pool._runner_states["m1"] is RunnerLifecycleState.WARM


class TestEvictSynchronously:
    def test_raises_on_non_warm_runner(self) -> None:
        pool = _make_pool()
        config = RunnerConfig(model_id="m1", memory_gb=2.0, load_time_s=1.0)
        pool.register(config)
        # State is COLD (default)
        with pytest.raises(RunnerStateError, match="non-WARM runner"):
            pool._evict_synchronously("m1")

    def test_transitions_warm_to_cold(self) -> None:
        pool = _make_pool(total_memory_gb=10.0)
        _register_warm(pool, "m1", memory_gb=3.0)
        pool._evict_synchronously("m1")
        assert pool._runner_states["m1"] is RunnerLifecycleState.COLD

    def test_releases_memory(self) -> None:
        pool = _make_pool(total_memory_gb=10.0)
        _register_warm(pool, "m1", memory_gb=3.0)
        assert pool._available_memory_gb == pytest.approx(7.0)
        pool._evict_synchronously("m1")
        assert pool._available_memory_gb == pytest.approx(10.0)
