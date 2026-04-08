"""Simulated model runner for the Helios test harness.

Provides a realistic runner with configurable load latency, inference latency,
and probabilistic failure injection. No GPU hardware required.
"""

import asyncio
import random

from helios.config import InferenceRequest, InferenceResult, RunnerConfig


class _RunnerLoadAttemptError(Exception):
    """Internal exception for a single load attempt failure.

    NOT a HeliosError -- must not cross the public API boundary.
    The router catches this via ``except Exception`` and retries up to max_retries.
    Only the router raises RunnerLoadError (public) after exhausting all attempts.
    """


class SimulatedModelRunner:
    """Three-method runner contract serving distinct lifecycle phases.

    - simulate_load: DOWNLOADING -> LOADING_TO_GPU (called under per-model lock)
    - infer: serves inference requests on a WARM/ACTIVE runner
    - is_healthy: health check for WARM runners only
    """

    def __init__(self, config: RunnerConfig, rng: random.Random | None = None) -> None:
        self._config = config
        self._rng = rng or random.Random()

    async def simulate_load(self) -> None:
        """Simulate DOWNLOADING -> LOADING_TO_GPU.

        Raises _RunnerLoadAttemptError probabilistically.
        Called by HeliosPool._simulate_load() under _runner_locks[model_id].
        Distinct from is_healthy(): this covers the load phase; is_healthy()
        covers WARM state. Both methods draw from self._rng.
        """
        await asyncio.sleep(self._config.load_time_s)
        if self._rng.random() < self._config.failure_rate:
            raise _RunnerLoadAttemptError(
                f"Simulated load failure for {self._config.model_id!r} "
                f"(failure_rate={self._config.failure_rate})"
            )

    async def infer(self, request: InferenceRequest) -> InferenceResult:
        """Simulate inference latency.

        Must use asyncio.sleep -- time.sleep blocks the event loop.
        """
        await asyncio.sleep(self._config.infer_time_s)
        return InferenceResult(
            model_id=self._config.model_id,
            result="[simulated output]",
            latency_ms=self._config.infer_time_s * 1000,
            cache_status="warm",
        )

    async def is_healthy(self) -> bool:
        """Health check for WARM runners only.

        ACTIVE runners are in the ACTIVE FSM state and are excluded from
        health checks -- see v11 section 12 design rationale.
        """
        return self._rng.random() >= self._config.failure_rate
