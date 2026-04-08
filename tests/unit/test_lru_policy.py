"""Unit tests for the LRU eviction policy."""

from helios.config import RunnerConfig, RunnerMetrics
from helios.policies.lru import LRUEvictionPolicy


def _config(model_id: str) -> RunnerConfig:
    return RunnerConfig(model_id=model_id, memory_gb=2.0, load_time_s=1.0)


def _metrics(model_id: str, last_request_at: float) -> RunnerMetrics:
    return RunnerMetrics(model_id=model_id, last_request_at=last_request_at)


class TestLRUEvictionPolicy:
    def test_evicts_oldest_runner(self) -> None:
        policy = LRUEvictionPolicy()
        configs = {m: _config(m) for m in ["a", "b", "c"]}
        metrics = {
            "a": _metrics("a", last_request_at=10.0),
            "b": _metrics("b", last_request_at=5.0),
            "c": _metrics("c", last_request_at=20.0),
        }
        assert policy.select_for_eviction(configs, metrics) == "b"

    def test_single_runner(self) -> None:
        policy = LRUEvictionPolicy()
        configs = {"x": _config("x")}
        metrics = {"x": _metrics("x", last_request_at=100.0)}
        assert policy.select_for_eviction(configs, metrics) == "x"

    def test_equal_timestamps_returns_deterministic_result(self) -> None:
        policy = LRUEvictionPolicy()
        configs = {m: _config(m) for m in ["a", "b", "c"]}
        metrics = {
            "a": _metrics("a", last_request_at=10.0),
            "b": _metrics("b", last_request_at=10.0),
            "c": _metrics("c", last_request_at=10.0),
        }
        # All equal — min() returns the first encountered, which is deterministic
        # for a given dict ordering (insertion order in Python 3.7+).
        result = policy.select_for_eviction(configs, metrics)
        assert result in {"a", "b", "c"}
        # Repeated calls must return the same result (deterministic).
        assert policy.select_for_eviction(configs, metrics) == result

    def test_most_recent_runner_not_evicted(self) -> None:
        policy = LRUEvictionPolicy()
        configs = {m: _config(m) for m in ["a", "b"]}
        metrics = {
            "a": _metrics("a", last_request_at=1.0),
            "b": _metrics("b", last_request_at=100.0),
        }
        assert policy.select_for_eviction(configs, metrics) != "b"

    def test_zero_timestamp_evicted_first(self) -> None:
        """Models with last_request_at=0.0 (never requested, e.g. pre-warmed)
        should be evicted before recently used models."""
        policy = LRUEvictionPolicy()
        configs = {m: _config(m) for m in ["prewarmed", "active"]}
        metrics = {
            "prewarmed": _metrics("prewarmed", last_request_at=0.0),
            "active": _metrics("active", last_request_at=50.0),
        }
        assert policy.select_for_eviction(configs, metrics) == "prewarmed"
