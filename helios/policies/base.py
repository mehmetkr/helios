"""Base eviction policy interface.

All eviction policies implement this ABC. The pool calls select_for_eviction()
under _memory_lock -- implementations must be synchronous and non-blocking.
"""

from abc import ABC, abstractmethod

from helios.config import RunnerConfig, RunnerMetrics


class BaseEvictionPolicy(ABC):
    @abstractmethod
    def select_for_eviction(
        self,
        configs: dict[str, RunnerConfig],
        metrics: dict[str, RunnerMetrics],
    ) -> str:
        """Return model_id of the WARM runner to evict.

        Called under _memory_lock -- must be synchronous and non-blocking.
        """
        ...
