"""Runner lifecycle finite state machine.

Defines the legal states and transitions for model runners.
Illegal transitions raise RunnerStateError.
"""

from enum import Enum, auto

from helios.exceptions import RunnerStateError

# Legal transitions: source -> set of allowed targets.
_TRANSITIONS: dict["RunnerLifecycleState", frozenset["RunnerLifecycleState"]] = {}


class RunnerLifecycleState(Enum):
    COLD = auto()
    DOWNLOADING = auto()
    LOADING_TO_GPU = auto()
    WARM = auto()
    ACTIVE = auto()
    EVICTING = auto()


# Define transitions after the enum exists.
_TRANSITIONS.update(
    {
        RunnerLifecycleState.COLD: frozenset({RunnerLifecycleState.DOWNLOADING}),
        RunnerLifecycleState.DOWNLOADING: frozenset({RunnerLifecycleState.LOADING_TO_GPU}),
        RunnerLifecycleState.LOADING_TO_GPU: frozenset(
            {RunnerLifecycleState.WARM, RunnerLifecycleState.COLD}
        ),
        RunnerLifecycleState.WARM: frozenset(
            {RunnerLifecycleState.ACTIVE, RunnerLifecycleState.EVICTING}
        ),
        RunnerLifecycleState.ACTIVE: frozenset(
            {RunnerLifecycleState.WARM, RunnerLifecycleState.COLD}
        ),
        RunnerLifecycleState.EVICTING: frozenset({RunnerLifecycleState.COLD}),
    }
)


def validate_transition(current: RunnerLifecycleState, target: RunnerLifecycleState) -> None:
    """Raise RunnerStateError if the transition is illegal.

    Scope: this is a specification/testing tool, not a runtime guard.
    HeliosPool does not call this function on every state mutation for
    performance reasons. Localized guards exist where correctness demands
    them: _evict_synchronously() verifies WARM before evicting, and
    dispatch() verifies WARM/ACTIVE before inferring. Use this function
    in tests to verify FSM correctness and in external consumers that
    want defensive validation.
    """
    allowed = _TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise RunnerStateError(
            f"Illegal transition: {current.name} -> {target.name}. "
            f"Allowed from {current.name}: "
            f"{', '.join(s.name for s in allowed) if allowed else 'none'}"
        )
