"""RequestRouter -- thin orchestration layer for retry, timeout, and admission.

Delegates all lifecycle operations to HeliosPool via its public API:
pool.ensure_loaded(), pool.dispatch(), pool.admit(), pool.release(), pool.config.
Never reaches into pool internals.
"""

from __future__ import annotations

import asyncio

from helios.config import InferenceRequest, InferenceResult
from helios.exceptions import (
    HeliosError,
    MemoryExhaustedError,
    RequestTimeoutError,
    RunnerLoadError,
)
from helios.pool import HeliosPool


class RequestRouter:
    """Orchestrates request routing with retry, timeout, and admission control.

    Owns the retry policy: retries load-phase failures up to max_retries,
    with brief backoff for transient MemoryExhaustedError. Dispatch errors
    (inference failures) propagate directly -- they are not retried and are
    not wrapped as RunnerLoadError.
    """

    def __init__(self, pool: HeliosPool) -> None:
        self._pool = pool

    async def route(self, request: InferenceRequest) -> InferenceResult:
        """Route a request through the pool.

        Admission -> load -> dispatch. The admit() call is OUTSIDE the try
        block so that a rejected admission does not trigger a spurious
        release(). The retry loop wraps only the load phase; dispatch() is
        called after the loop exception block, so inference errors propagate
        with their original typed exceptions.
        """
        self._pool.admit()  # raises AdmissionRejectedError if at capacity
        try:
            was_warm = False
            for attempt in range(self._pool.config.max_retries + 1):
                try:
                    was_warm = await asyncio.wait_for(
                        self._pool.ensure_loaded(request.model_id),
                        timeout=self._pool.config.request_timeout_s,
                    )
                except TimeoutError as exc:
                    raise RequestTimeoutError(
                        f"Model {request.model_id!r} not warm after "
                        f"{self._pool.config.request_timeout_s}s"
                    ) from exc
                except MemoryExhaustedError as exc:
                    if exc.permanent:
                        raise  # structural -- not retryable
                    if attempt == self._pool.config.max_retries:
                        raise  # transient but out of retries
                    # Transient: configurable backoff, then retry.
                    await asyncio.sleep(self._pool.config.retry_backoff_s)
                    continue
                except HeliosError:
                    raise  # RunnerStateError, AdmissionRejectedError, etc. -- not retryable
                except Exception as exc:
                    # Catches _RunnerLoadAttemptError (internal, not HeliosError)
                    # and other unexpected errors from the load path.
                    if attempt == self._pool.config.max_retries:
                        raise RunnerLoadError(
                            f"Runner for {request.model_id!r} failed to load "
                            f"after {self._pool.config.max_retries + 1} attempts"
                        ) from exc
                    # Retry: _initiate_load_and_resolve.finally already cleaned
                    # up the failed future. Next ensure_loaded() starts fresh.
                    continue
                # Load succeeded -- dispatch is outside the retry try/except
                # scope. Inference errors propagate directly to the caller,
                # not wrapped as RunnerLoadError. Preserves typed exceptions.
                result = await self._pool.dispatch(request)
                if not was_warm:
                    result = result.model_copy(update={"cache_status": "cold"})
                return result
            # Should be unreachable -- loop either returns or raises.
            raise RunnerLoadError(f"Runner for {request.model_id!r} failed to load")
        finally:
            self._pool.release()
