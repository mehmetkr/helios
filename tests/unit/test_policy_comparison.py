"""Layer 1: Head-to-head eviction policy comparison.

Proves that LRU and cost-based policies make different eviction decisions
under the same inputs -- the core value proposition of cost-based eviction.

Scenario: two WARM models compete for eviction.
- "popular": older last_request_at BUT high predicted_demand (0.9)
- "recent": newer last_request_at BUT low predicted_demand (0.0)

LRU only sees recency -> evicts "popular" (oldest).
Cost-based sees demand -> protects "popular" (high restart penalty), evicts "recent".
"""

import time

from helios.config import RunnerConfig, RunnerMetrics
from helios.policies.cost_based import CostBasedEvictionPolicy
from helios.policies.lru import LRUEvictionPolicy

TOTAL_MEMORY = 80.0


def _make_scenario() -> tuple[dict[str, RunnerConfig], dict[str, RunnerMetrics]]:
    """Create the divergent scenario: same configs/metrics, different optimal victims."""
    now = time.monotonic()
    configs = {
        "popular": RunnerConfig(model_id="popular", memory_gb=4.0, load_time_s=10.0),
        "recent": RunnerConfig(model_id="recent", memory_gb=4.0, load_time_s=10.0),
    }
    metrics = {
        "popular": RunnerMetrics(
            model_id="popular",
            last_request_at=now - 300.0,  # old -- LRU target
            request_rate_per_min=20.0,
            predicted_demand=0.9,  # high -- cost-based protects
        ),
        "recent": RunnerMetrics(
            model_id="recent",
            last_request_at=now - 5.0,  # recent -- LRU spares
            request_rate_per_min=0.5,
            predicted_demand=0.0,  # low -- cost-based evicts
        ),
    }
    return configs, metrics


class TestPolicyComparison:
    def test_lru_evicts_oldest_regardless_of_demand(self) -> None:
        """LRU ignores demand -- evicts the model with oldest last_request_at."""
        configs, metrics = _make_scenario()
        victim = LRUEvictionPolicy().select_for_eviction(configs, metrics)
        assert victim == "popular"

    def test_cost_based_protects_high_demand_model(self) -> None:
        """Cost-based protects the high-demand model and evicts the low-demand one."""
        configs, metrics = _make_scenario()
        victim = CostBasedEvictionPolicy(TOTAL_MEMORY).select_for_eviction(configs, metrics)
        assert victim == "recent"

    def test_policies_disagree_on_victim(self) -> None:
        """Under the same inputs, LRU and cost-based choose different victims.

        This is the core value proposition: cost-based eviction preserves
        models that are expensive to reload and likely to be needed soon,
        even if they haven't been requested recently.
        """
        configs, metrics = _make_scenario()
        lru_victim = LRUEvictionPolicy().select_for_eviction(configs, metrics)
        cost_victim = CostBasedEvictionPolicy(TOTAL_MEMORY).select_for_eviction(configs, metrics)
        assert lru_victim != cost_victim
        assert lru_victim == "popular"  # LRU: oldest
        assert cost_victim == "recent"  # cost-based: lowest demand
