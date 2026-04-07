"""Unit tests for the runner lifecycle FSM."""

import pytest

from helios.exceptions import RunnerStateError
from helios.fsm import RunnerLifecycleState, validate_transition


class TestLegalTransitions:
    """Every legal transition in the FSM should pass without error."""

    @pytest.mark.parametrize(
        ("source", "target"),
        [
            (RunnerLifecycleState.COLD, RunnerLifecycleState.DOWNLOADING),
            (RunnerLifecycleState.DOWNLOADING, RunnerLifecycleState.LOADING_TO_GPU),
            (RunnerLifecycleState.LOADING_TO_GPU, RunnerLifecycleState.WARM),
            (RunnerLifecycleState.LOADING_TO_GPU, RunnerLifecycleState.COLD),
            (RunnerLifecycleState.WARM, RunnerLifecycleState.ACTIVE),
            (RunnerLifecycleState.WARM, RunnerLifecycleState.EVICTING),
            (RunnerLifecycleState.ACTIVE, RunnerLifecycleState.WARM),
            (RunnerLifecycleState.ACTIVE, RunnerLifecycleState.COLD),
            (RunnerLifecycleState.EVICTING, RunnerLifecycleState.COLD),
        ],
    )
    def test_legal_transition(
        self, source: RunnerLifecycleState, target: RunnerLifecycleState
    ) -> None:
        validate_transition(source, target)


class TestIllegalTransitions:
    """Illegal transitions must raise RunnerStateError."""

    @pytest.mark.parametrize(
        ("source", "target"),
        [
            (RunnerLifecycleState.COLD, RunnerLifecycleState.WARM),
            (RunnerLifecycleState.COLD, RunnerLifecycleState.ACTIVE),
            (RunnerLifecycleState.COLD, RunnerLifecycleState.EVICTING),
            (RunnerLifecycleState.DOWNLOADING, RunnerLifecycleState.WARM),
            (RunnerLifecycleState.DOWNLOADING, RunnerLifecycleState.COLD),
            (RunnerLifecycleState.WARM, RunnerLifecycleState.COLD),
            (RunnerLifecycleState.WARM, RunnerLifecycleState.DOWNLOADING),
            (RunnerLifecycleState.ACTIVE, RunnerLifecycleState.EVICTING),
            (RunnerLifecycleState.ACTIVE, RunnerLifecycleState.DOWNLOADING),
            (RunnerLifecycleState.EVICTING, RunnerLifecycleState.WARM),
            (RunnerLifecycleState.EVICTING, RunnerLifecycleState.ACTIVE),
        ],
    )
    def test_illegal_transition(
        self, source: RunnerLifecycleState, target: RunnerLifecycleState
    ) -> None:
        with pytest.raises(RunnerStateError, match=f"{source.name} -> {target.name}"):
            validate_transition(source, target)


class TestInitialState:
    """All runners begin in COLD state."""

    def test_cold_is_defined(self) -> None:
        assert RunnerLifecycleState.COLD is not None

    def test_all_states_exist(self) -> None:
        expected = {"COLD", "DOWNLOADING", "LOADING_TO_GPU", "WARM", "ACTIVE", "EVICTING"}
        actual = {s.name for s in RunnerLifecycleState}
        assert actual == expected
