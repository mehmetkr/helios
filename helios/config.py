"""Helios configuration and data models.

Static configuration and dynamic runtime metrics are strictly separated.
Internal data structures use frozen dataclasses; API-facing models use
Pydantic v2 for validation and serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

# --- Internal data structures (no Pydantic overhead) ---


@dataclass(frozen=True)
class RunnerConfig:
    """Immutable model properties -- set at registration, never changed."""

    model_id: str
    memory_gb: float
    load_time_s: float
    failure_rate: float = 0.0
    # Simulation-only. Probability of failure during LOADING_TO_GPU state.
    # Used exclusively by SimulatedModelRunner to inject failure on the retry path.
    # Not an eviction factor -- CostBasedEvictionPolicy does not read this field.
    infer_time_s: float = 0.05
    # Simulation-only. Latency for SimulatedModelRunner.infer().

    def __post_init__(self) -> None:
        if self.memory_gb <= 0:
            raise ValueError(f"memory_gb must be > 0, got {self.memory_gb}")
        if self.load_time_s <= 0:
            raise ValueError(f"load_time_s must be > 0, got {self.load_time_s}")
        if not (0.0 <= self.failure_rate <= 1.0):
            raise ValueError(f"failure_rate must be in [0.0, 1.0], got {self.failure_rate}")
        if self.infer_time_s <= 0:
            raise ValueError(f"infer_time_s must be > 0, got {self.infer_time_s}")


@dataclass
class RunnerMetrics:
    """Dynamic runtime state -- updated on every request and pre-warm tick."""

    model_id: str
    last_request_at: float = field(default_factory=lambda: 0.0)
    request_rate_per_min: float = 0.0  # EWMA, see Pre-warm Loop
    predicted_demand: float = 0.0  # normalized [0.0, 1.0] from HoltPredictor


@dataclass(frozen=True)
class PoolConfig:
    """All Helios pool configuration. All fields have documented defaults."""

    total_memory_gb: float = 80.0
    # Hard memory budget. Sum of all runners in GPU_MEMORY_STATES must never exceed this.

    max_concurrent_loads: int = 4
    # Maximum simultaneous model loads. Enforced by asyncio.Semaphore.

    request_timeout_s: float = 30.0
    # Maximum wait for a WARM runner before RequestTimeoutError is raised.

    prewarm_threshold: float = 0.3
    # Normalized demand forecast in [0.0, 1.0] above which a COLD model is proactively
    # pre-warmed. Directly comparable to HoltPredictor.predict_next() output.
    # 0.3 = pre-warm when predicted demand exceeds 30% of that model's historical peak.

    prewarm_interval_s: float = 60.0
    # Interval at which the pre-warm loop runs and feeds HoltPredictor.record().

    ewma_alpha: float = 0.3
    # Smoothing factor for request_rate_per_min EWMA. Higher values weight
    # recent observations more heavily. Range: (0.0, 1.0).

    max_retries: int = 2
    # Load retry attempts per request before RunnerLoadError is raised.

    health_check_interval_s: float = 30.0
    # Interval at which WARM runners are health-checked via is_healthy().
    # ACTIVE runners are excluded -- see health check design rationale.

    max_queued_requests: int = 100
    # Maximum concurrent route() calls. Prevents unbounded queue growth under
    # thundering-herd conditions. AdmissionRejectedError raised when exceeded.

    retry_backoff_s: float = 0.1
    # Backoff duration between retries for transient MemoryExhaustedError.
    # Used by RequestRouter when a load fails due to temporary memory pressure.

    def __post_init__(self) -> None:
        """Enforce range constraints that the type system cannot express."""
        if not (0.0 <= self.prewarm_threshold <= 1.0):
            raise ValueError(
                f"prewarm_threshold must be in [0.0, 1.0], got {self.prewarm_threshold}. "
                f"Values > 1.0 would never match the predictor's normalized output."
            )
        if not (0.0 < self.ewma_alpha < 1.0):
            raise ValueError(f"ewma_alpha must be in (0.0, 1.0), got {self.ewma_alpha}.")
        if self.total_memory_gb <= 0:
            raise ValueError(f"total_memory_gb must be positive, got {self.total_memory_gb}.")
        if self.max_concurrent_loads < 1:
            raise ValueError(
                f"max_concurrent_loads must be >= 1, got {self.max_concurrent_loads}."
            )
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be >= 0 (0 = try once, no retries), "
                f"got {self.max_retries}. "
                f"Negative values cause range(0) in the retry loop, silently "
                f"skipping all load attempts."
            )
        if self.max_queued_requests < 1:
            raise ValueError(f"max_queued_requests must be >= 1, got {self.max_queued_requests}.")
        if self.retry_backoff_s < 0:
            raise ValueError(f"retry_backoff_s must be >= 0, got {self.retry_backoff_s}.")
        # Interval fields: zero or negative values would cause asyncio.sleep
        # to behave incorrectly (sleep(0) yields once but negative is undefined).
        for field_name, value in (
            ("request_timeout_s", self.request_timeout_s),
            ("prewarm_interval_s", self.prewarm_interval_s),
            ("health_check_interval_s", self.health_check_interval_s),
        ):
            if value <= 0:
                raise ValueError(f"{field_name} must be positive (> 0), got {value}.")


# --- API layer models (Pydantic v2) ---


class InferenceRequest(BaseModel):
    """Incoming inference request from the HTTP layer."""

    model_id: str
    payload: str
    priority: int = Field(default=0, ge=0, le=10)


class InferenceResult(BaseModel):
    """Response returned to the HTTP layer after inference."""

    model_id: str
    result: str
    latency_ms: float
    cache_status: str  # "warm" | "cold"
