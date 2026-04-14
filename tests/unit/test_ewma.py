"""Layer 1: EWMA rate normalization formula correctness tests.

Pure synchronous tests verifying the formula:
    rate_new = alpha * (count * 60.0 / interval_s) + (1 - alpha) * rate_old

Tested at boundary alpha values, different tick intervals, steady-state
convergence, and zero-count decay.
"""

import pytest


def ewma_update(
    alpha: float,
    count: float,
    interval_s: float,
    rate_old: float,
) -> float:
    """Replicate the EWMA formula from HeliosPool._prewarm_loop."""
    return alpha * (count * 60.0 / interval_s) + (1.0 - alpha) * rate_old


class TestNormalization:
    """The x 60.0 / interval_s term converts raw count to per-minute rate."""

    def test_default_interval_60s(self) -> None:
        """With interval=60s, count IS the per-minute rate directly."""
        rate = ewma_update(alpha=0.3, count=10, interval_s=60.0, rate_old=0.0)
        assert rate == pytest.approx(3.0)  # 0.3 * (10 * 60/60) = 0.3 * 10

    def test_short_interval_1s(self) -> None:
        """With interval=1s, 1 request/tick = 60 rpm."""
        rate = ewma_update(alpha=0.3, count=1, interval_s=1.0, rate_old=0.0)
        assert rate == pytest.approx(18.0)  # 0.3 * (1 * 60/1) = 0.3 * 60

    def test_different_intervals_same_true_rate(self) -> None:
        """Same true rate (60 rpm) measured at different intervals converges
        to the same steady-state value."""
        # 60 rpm = 60 requests per 60s tick = 1 request per 1s tick
        rate_60s = 0.0
        rate_1s = 0.0
        for _ in range(200):
            rate_60s = ewma_update(alpha=0.3, count=60, interval_s=60.0, rate_old=rate_60s)
            rate_1s = ewma_update(alpha=0.3, count=1, interval_s=1.0, rate_old=rate_1s)
        assert rate_60s == pytest.approx(60.0, abs=0.1)
        assert rate_1s == pytest.approx(60.0, abs=0.1)


class TestBoundaryAlpha:
    """Alpha near 0 adapts slowly; alpha near 1 adapts immediately."""

    def test_alpha_near_zero_barely_moves(self) -> None:
        """Alpha=0.01: new observation barely affects the rate."""
        rate = ewma_update(alpha=0.01, count=100, interval_s=60.0, rate_old=0.0)
        assert rate == pytest.approx(1.0)  # 0.01 * 100 = 1.0

    def test_alpha_near_zero_barely_decays(self) -> None:
        """Alpha=0.01: rate decays very slowly on zero count."""
        rate = ewma_update(alpha=0.01, count=0, interval_s=60.0, rate_old=100.0)
        assert rate == pytest.approx(99.0)  # 0.99 * 100

    def test_alpha_near_one_jumps_immediately(self) -> None:
        """Alpha=0.99: rate jumps to match the new observation."""
        rate = ewma_update(alpha=0.99, count=100, interval_s=60.0, rate_old=0.0)
        assert rate == pytest.approx(99.0)  # 0.99 * 100

    def test_alpha_near_one_drops_immediately(self) -> None:
        """Alpha=0.99: rate drops immediately on zero count."""
        rate = ewma_update(alpha=0.99, count=0, interval_s=60.0, rate_old=100.0)
        assert rate == pytest.approx(1.0)  # 0.01 * 100


class TestSteadyStateConvergence:
    """Constant count per tick converges to the correct per-minute rate."""

    def test_converges_to_true_rate(self) -> None:
        """10 requests per 60s tick → converges to 10 rpm."""
        rate = 0.0
        for _ in range(100):
            rate = ewma_update(alpha=0.3, count=10, interval_s=60.0, rate_old=rate)
        assert rate == pytest.approx(10.0, abs=0.01)

    def test_converges_with_short_interval(self) -> None:
        """5 requests per 1s tick → converges to 300 rpm."""
        rate = 0.0
        for _ in range(100):
            rate = ewma_update(alpha=0.3, count=5, interval_s=1.0, rate_old=rate)
        assert rate == pytest.approx(300.0, abs=0.1)


class TestZeroCountDecay:
    """Sustained zero counts cause rate to decay toward 0."""

    def test_decays_to_near_zero(self) -> None:
        """Starting at rate=100, zero counts decay exponentially."""
        rate = 100.0
        for _ in range(50):
            rate = ewma_update(alpha=0.3, count=0, interval_s=60.0, rate_old=rate)
        assert rate < 0.01

    def test_decay_rate_matches_formula(self) -> None:
        """Each zero-count tick multiplies rate by (1 - alpha)."""
        rate = 100.0
        alpha = 0.3
        for _ in range(5):
            rate = ewma_update(alpha=alpha, count=0, interval_s=60.0, rate_old=rate)
        expected = 100.0 * (1.0 - alpha) ** 5
        assert rate == pytest.approx(expected)
