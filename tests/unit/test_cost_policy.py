"""Unit tests for the cost-based eviction policy.

Tests all 5 factors: load_time_s, memory_gb, time_since_last_request,
request_rate_per_min, and predicted_demand. Plus edge cases.
"""

import time
from unittest.mock import patch

from helios.config import RunnerConfig, RunnerMetrics
from helios.policies.cost_based import CostBasedEvictionPolicy

TOTAL_MEMORY = 80.0


def _config(model_id: str, *, memory_gb: float = 4.0, load_time_s: float = 5.0) -> RunnerConfig:
    return RunnerConfig(model_id=model_id, memory_gb=memory_gb, load_time_s=load_time_s)


def _metrics(
    model_id: str,
    *,
    last_request_at: float = 0.0,
    request_rate_per_min: float = 0.0,
    predicted_demand: float = 0.0,
) -> RunnerMetrics:
    return RunnerMetrics(
        model_id=model_id,
        last_request_at=last_request_at,
        request_rate_per_min=request_rate_per_min,
        predicted_demand=predicted_demand,
    )


class TestCostBasedEvictionPolicy:
    """All five factors in the eviction score formula."""

    def test_idle_model_evicted_over_active(self) -> None:
        """Higher idle_time increases eviction score."""
        policy = CostBasedEvictionPolicy(TOTAL_MEMORY)
        now = time.monotonic()
        configs = {m: _config(m) for m in ["idle", "active"]}
        metrics = {
            "idle": _metrics("idle", last_request_at=now - 1000),
            "active": _metrics("active", last_request_at=now - 1),
        }
        assert policy.select_for_eviction(configs, metrics) == "idle"

    def test_low_demand_evicted_over_high_demand(self) -> None:
        """Higher predicted_demand increases restart_penalty, lowering score."""
        policy = CostBasedEvictionPolicy(TOTAL_MEMORY)
        now = time.monotonic()
        configs = {m: _config(m) for m in ["cold", "hot"]}
        metrics = {
            "cold": _metrics("cold", last_request_at=now - 100, predicted_demand=0.0),
            "hot": _metrics("hot", last_request_at=now - 100, predicted_demand=0.9),
        }
        assert policy.select_for_eviction(configs, metrics) == "cold"

    def test_small_model_protected_over_large(self) -> None:
        """Larger memory_gb increases memory_pressure, lowering score (protected)."""
        policy = CostBasedEvictionPolicy(TOTAL_MEMORY)
        now = time.monotonic()
        configs = {
            "small": _config("small", memory_gb=2.0),
            "large": _config("large", memory_gb=8.0),
        }
        metrics = {
            "small": _metrics("small", last_request_at=now - 100),
            "large": _metrics("large", last_request_at=now - 100),
        }
        # Small model has lower memory_pressure -> higher score -> evicted first
        assert policy.select_for_eviction(configs, metrics) == "small"

    def test_fast_loading_model_evicted_over_slow(self) -> None:
        """Higher load_time_s increases restart_penalty, lowering score (protected)."""
        policy = CostBasedEvictionPolicy(TOTAL_MEMORY)
        now = time.monotonic()
        configs = {
            "fast": _config("fast", load_time_s=1.0),
            "slow": _config("slow", load_time_s=10.0),
        }
        metrics = {
            "fast": _metrics("fast", last_request_at=now - 100),
            "slow": _metrics("slow", last_request_at=now - 100),
        }
        assert policy.select_for_eviction(configs, metrics) == "fast"

    def test_low_rate_evicted_over_high_rate(self) -> None:
        """Higher request_rate_per_min lowers recency_weight, lowering score."""
        policy = CostBasedEvictionPolicy(TOTAL_MEMORY)
        now = time.monotonic()
        configs = {m: _config(m) for m in ["quiet", "busy"]}
        metrics = {
            "quiet": _metrics("quiet", last_request_at=now - 100, request_rate_per_min=0.0),
            "busy": _metrics("busy", last_request_at=now - 100, request_rate_per_min=60.0),
        }
        assert policy.select_for_eviction(configs, metrics) == "quiet"

    def test_single_runner(self) -> None:
        """Single candidate must be returned."""
        policy = CostBasedEvictionPolicy(TOTAL_MEMORY)
        configs = {"only": _config("only")}
        metrics = {"only": _metrics("only")}
        assert policy.select_for_eviction(configs, metrics) == "only"

    def test_equal_scores_returns_deterministic_result(self) -> None:
        """Equal metrics across runners should return a deterministic result."""
        policy = CostBasedEvictionPolicy(TOTAL_MEMORY)
        now = time.monotonic()
        configs = {m: _config(m) for m in ["a", "b", "c"]}
        metrics = {m: _metrics(m, last_request_at=now - 50) for m in ["a", "b", "c"]}
        result = policy.select_for_eviction(configs, metrics)
        assert result in {"a", "b", "c"}
        assert policy.select_for_eviction(configs, metrics) == result

    def test_maximum_memory_pressure(self) -> None:
        """A model using the entire memory budget has memory_pressure=1.0."""
        policy = CostBasedEvictionPolicy(TOTAL_MEMORY)
        now = time.monotonic()
        configs = {
            "full": _config("full", memory_gb=TOTAL_MEMORY),
            "tiny": _config("tiny", memory_gb=1.0),
        }
        metrics = {
            "full": _metrics("full", last_request_at=now - 100),
            "tiny": _metrics("tiny", last_request_at=now - 100),
        }
        # tiny has lower memory_pressure -> higher score -> evicted
        assert policy.select_for_eviction(configs, metrics) == "tiny"

    def test_minimum_predicted_demand(self) -> None:
        """predicted_demand=0.0 should not cause division by zero (epsilon guards)."""
        policy = CostBasedEvictionPolicy(TOTAL_MEMORY)
        now = time.monotonic()
        configs = {"zero_demand": _config("zero_demand")}
        metrics = {
            "zero_demand": _metrics("zero_demand", last_request_at=now - 10, predicted_demand=0.0),
        }
        # Should not raise
        result = policy.select_for_eviction(configs, metrics)
        assert result == "zero_demand"

    @patch("helios.policies.cost_based.time.monotonic", return_value=1000.0)
    def test_score_formula_correctness(self, _mock_time: object) -> None:
        """Verify the formula produces expected relative ordering."""
        policy = CostBasedEvictionPolicy(total_memory_gb=100.0)
        configs = {
            "a": _config("a", memory_gb=10.0, load_time_s=2.0),
            "b": _config("b", memory_gb=5.0, load_time_s=8.0),
        }
        metrics = {
            "a": _metrics(
                "a", last_request_at=900.0, request_rate_per_min=10.0, predicted_demand=0.5
            ),
            "b": _metrics(
                "b", last_request_at=500.0, request_rate_per_min=1.0, predicted_demand=0.1
            ),
        }
        # Model b: idle_time=500, recency=1/2=0.5, restart=8*0.100001=0.800008, pressure=0.05
        # Score b = (500 * 0.5) / (0.800008 * 0.05) = 250 / 0.040000 = 6250
        # Model a: idle_time=100, recency=1/11≈0.0909, restart=2*0.500001=1.000002, pressure=0.1
        # Score a = (100 * 0.0909) / (1.000002 * 0.1) = 9.09 / 0.100 = 90.9
        # b has higher score -> b is evicted
        assert policy.select_for_eviction(configs, metrics) == "b"
