"""Helios exception hierarchy.

All public Helios exceptions live in this module. A shared base class allows
callers to catch any Helios error generically while enabling specific handling
where needed. Internal private exceptions (prefixed with _) may be defined in
their owning module.
"""


class HeliosError(Exception):
    """Base exception for all Helios errors."""


class RunnerLoadError(HeliosError):
    """Model runner failed to load after max_retries attempts."""


class MemoryExhaustedError(HeliosError):
    """Memory budget cannot accommodate a new model.

    Either required_gb > total_memory_gb (structural), or
    all eviction candidates are ACTIVE (transient).
    """

    def __init__(self, message: str, *, permanent: bool = False) -> None:
        super().__init__(message)
        self.permanent = permanent


class RunnerStateError(HeliosError):
    """Illegal state transition attempted on a runner FSM."""


class RequestTimeoutError(HeliosError):
    """Request exceeded request_timeout_s waiting for a WARM runner."""


class AdmissionRejectedError(HeliosError):
    """Pool cannot accept request -- at capacity with no buffer headroom."""
