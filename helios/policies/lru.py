"""LRU (Least Recently Used) eviction policy.

Evicts the runner with the oldest last_request_at. Synchronous, O(n).
"""

from helios.config import RunnerConfig, RunnerMetrics
from helios.policies.base import BaseEvictionPolicy


class LRUEvictionPolicy(BaseEvictionPolicy):
    def select_for_eviction(
        self,
        configs: dict[str, RunnerConfig],
        metrics: dict[str, RunnerMetrics],
    ) -> str:
        return min(metrics.values(), key=lambda m: m.last_request_at).model_id
