"""Unit tests for SimulatedModelRunner -- synchronous contract checks (Layer 1).

Async methods (simulate_load, infer, is_healthy) are tested in Layer 2.
"""

import random

from helios.config import RunnerConfig
from helios.simulation.runner import SimulatedModelRunner, _RunnerLoadAttemptError


class TestRunnerLoadAttemptError:
    def test_is_not_helios_error(self) -> None:
        """_RunnerLoadAttemptError must not cross the public API boundary.

        The router's retry loop relies on this exception NOT being caught
        by ``except HeliosError: raise``. If someone changes the inheritance,
        the retry loop silently breaks.
        """
        from helios.exceptions import HeliosError

        assert not issubclass(_RunnerLoadAttemptError, HeliosError)

    def test_is_exception_subclass(self) -> None:
        assert issubclass(_RunnerLoadAttemptError, Exception)


class TestSimulatedModelRunnerInit:
    def test_default_rng(self) -> None:
        config = RunnerConfig(model_id="m1", memory_gb=2.0, load_time_s=1.0)
        runner = SimulatedModelRunner(config)
        assert runner._rng is not None
        assert isinstance(runner._rng, random.Random)

    def test_explicit_rng(self) -> None:
        config = RunnerConfig(model_id="m1", memory_gb=2.0, load_time_s=1.0)
        rng = random.Random(42)
        runner = SimulatedModelRunner(config, rng=rng)
        assert runner._rng is rng

    def test_stores_config(self) -> None:
        config = RunnerConfig(model_id="m1", memory_gb=2.0, load_time_s=1.0)
        runner = SimulatedModelRunner(config)
        assert runner._config is config
