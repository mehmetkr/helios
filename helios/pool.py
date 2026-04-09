"""HeliosPool -- core pool manager with two-phase locking and iterative eviction.

Manages the lifecycle of simulated model runners: registration, memory budget,
eviction, load orchestration, pre-warming, health checks, and graceful shutdown.
This module is built incrementally -- load orchestration, pre-warm loop, health
checks, and shutdown are added in subsequent commits.
"""

from __future__ import annotations

import asyncio
import random
from collections import defaultdict
from collections.abc import Coroutine
from typing import Any

import structlog

from helios.config import PoolConfig, RunnerConfig, RunnerMetrics
from helios.exceptions import (
    AdmissionRejectedError,
    HeliosError,
    MemoryExhaustedError,
    RunnerStateError,
)
from helios.fsm import RunnerLifecycleState
from helios.observability.metrics import EVICTIONS_TOTAL
from helios.policies.base import BaseEvictionPolicy
from helios.policies.lru import LRUEvictionPolicy
from helios.prediction.holt import HoltPredictor
from helios.simulation.runner import SimulatedModelRunner

logger = structlog.get_logger(__name__)


class HeliosPool:
    """Intelligent control plane for serverless GPU inference.

    Two-phase locking (deadlock-free):
    - _memory_lock (pool-wide, brief): budget arithmetic only, no I/O
    - _runner_locks[model_id] (per-model, long): state transitions during load
    The two locks are never held simultaneously.
    """

    def __init__(
        self,
        config: PoolConfig | None = None,
        eviction_policy: BaseEvictionPolicy | None = None,
    ) -> None:
        self._config = config or PoolConfig()
        self._eviction_policy = eviction_policy or LRUEvictionPolicy()

        # --- Concurrency primitives ---
        self._memory_lock = asyncio.Lock()
        self._runner_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._load_semaphore = asyncio.Semaphore(self._config.max_concurrent_loads)

        # --- Memory budget ---
        self._available_memory_gb: float = self._config.total_memory_gb

        # --- Model registries (populated by register()) ---
        self._configs: dict[str, RunnerConfig] = {}
        self._metrics: dict[str, RunnerMetrics] = {}
        self._runner_states: dict[str, RunnerLifecycleState] = {}
        self._predictors: dict[str, HoltPredictor] = {}
        self._runners: dict[str, SimulatedModelRunner] = {}

        # --- Runtime state ---
        self._active_requests: defaultdict[str, int] = defaultdict(int)
        self._request_counter: defaultdict[str, int] = defaultdict(int)
        self._load_futures: dict[str, asyncio.Future[None]] = {}
        self._in_flight_requests: int = 0
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._started: bool = False

    # --- Public API ---

    @property
    def config(self) -> PoolConfig:
        """Public read-only access to pool configuration."""
        return self._config

    def register(self, config: RunnerConfig, rng: random.Random | None = None) -> None:
        """Register a model runner. Must be called before start().

        Populates all internal registries and creates a SimulatedModelRunner.
        All runners start in COLD state.
        """
        if self._started:
            raise HeliosError("pool is live -- registration is closed")
        self._configs[config.model_id] = config
        self._metrics[config.model_id] = RunnerMetrics(model_id=config.model_id)
        self._runner_states[config.model_id] = RunnerLifecycleState.COLD
        self._predictors[config.model_id] = HoltPredictor(config.model_id)
        self._runners[config.model_id] = SimulatedModelRunner(config, rng=rng)

    def start(self) -> None:
        """Start the pool's background tasks.

        Must be called from within a running event loop (e.g., inside an async
        function or FastAPI lifespan). Idempotent: returns immediately if
        already started. Background task spawning is added in later commits.
        """
        if self._started:
            return
        self._started = True

    def admit(self) -> None:
        """Admission check -- raises AdmissionRejectedError if at capacity.

        Called by RequestRouter at request entry. Check-and-increment is
        atomic (no await between them).
        """
        if self._in_flight_requests >= self._config.max_queued_requests:
            raise AdmissionRejectedError(
                f"Pool at capacity ({self._config.max_queued_requests} queued requests)"
            )
        self._in_flight_requests += 1

    def release(self) -> None:
        """Decrement admission counter. Called by RequestRouter in finally block."""
        self._in_flight_requests -= 1

    # --- Memory management (called under _memory_lock) ---

    def _evict_until_free(self, required_gb: float) -> None:
        """Iteratively evict WARM runners until required_gb is available.

        Called under _memory_lock. Never acquires per-model locks.

        Two termination conditions raise MemoryExhaustedError:
        1. Structural impossibility: required_gb > total_memory_gb (permanent)
        2. Transient impossibility: no WARM runners available (not permanent)
        """
        if required_gb > self._config.total_memory_gb:
            raise MemoryExhaustedError(
                f"Model requires {required_gb:.1f}GB but total budget is "
                f"{self._config.total_memory_gb:.1f}GB -- structurally impossible.",
                permanent=True,
            )

        while self._available_memory_gb < required_gb:
            warm_candidates = {
                mid: self._configs[mid]
                for mid, state in self._runner_states.items()
                if state is RunnerLifecycleState.WARM
            }
            if not warm_candidates:
                raise MemoryExhaustedError(
                    f"Cannot free {required_gb:.1f}GB: no WARM runners available. "
                    f"All runners may be ACTIVE or LOADING.",
                    permanent=False,
                )
            victim_id = self._eviction_policy.select_for_eviction(
                warm_candidates,
                {mid: self._metrics[mid] for mid in warm_candidates},
            )
            self._evict_synchronously(victim_id)

    def _evict_synchronously(self, model_id: str) -> None:
        """Transition a WARM runner through EVICTING -> COLD atomically.

        Called exclusively under _memory_lock. Because ACTIVE runners are
        immune to eviction, all callers are guaranteed to receive WARM runners
        only. WARM runners have no in-flight requests, so no draining is
        needed -- the double transition is instantaneous in the simulation.
        """
        if self._runner_states[model_id] is not RunnerLifecycleState.WARM:
            raise RunnerStateError(
                f"_evict_synchronously called on non-WARM runner: "
                f"{model_id} is {self._runner_states[model_id].name}"
            )
        self._runner_states[model_id] = RunnerLifecycleState.EVICTING
        self._runner_states[model_id] = RunnerLifecycleState.COLD
        self._available_memory_gb += self._configs[model_id].memory_gb
        EVICTIONS_TOTAL.labels(
            model_id=model_id, reason=self._eviction_policy.__class__.__name__
        ).inc()

    def _reserve_memory(self, model_id: str) -> None:
        """Decrement available memory for a model about to load.

        Called under _memory_lock after _evict_until_free succeeds.
        """
        self._available_memory_gb -= self._configs[model_id].memory_gb

    def _release_memory(self, model_id: str) -> None:
        """Increment available memory when a model is evicted or load fails.

        Called under _memory_lock.
        """
        self._available_memory_gb += self._configs[model_id].memory_gb

    # --- Background task management ---

    def _spawn_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        """Create a tracked background task. Auto-discards on completion."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task
