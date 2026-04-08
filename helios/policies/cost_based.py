"""Cost-based eviction policy with five-factor scoring.

Considers load time, memory pressure, idle time, request rate, and predicted
demand to select the optimal eviction candidate. All factors are dimensionally
consistent.
"""

import time

from helios.config import RunnerConfig, RunnerMetrics
from helios.policies.base import BaseEvictionPolicy


class CostBasedEvictionPolicy(BaseEvictionPolicy):
    def __init__(self, total_memory_gb: float) -> None:
        self._total_memory_gb = total_memory_gb

    def select_for_eviction(
        self,
        configs: dict[str, RunnerConfig],
        metrics: dict[str, RunnerMetrics],
    ) -> str:
        now = time.monotonic()

        def eviction_score(model_id: str) -> float:
            cfg = configs[model_id]
            met = metrics[model_id]
            # predicted_demand is normalized [0,1] -- all terms dimensionless
            idle_time = now - met.last_request_at  # seconds
            # epsilon prevents div-by-zero when predicted_demand is 0.0
            restart_penalty = cfg.load_time_s * (met.predicted_demand + 1e-6)
            recency_weight = 1.0 / (1.0 + met.request_rate_per_min)  # dimensionless
            memory_pressure = cfg.memory_gb / self._total_memory_gb  # dimensionless [0,1]
            return (idle_time * recency_weight) / (restart_penalty * memory_pressure)

        return max(configs.keys(), key=eviction_score)
