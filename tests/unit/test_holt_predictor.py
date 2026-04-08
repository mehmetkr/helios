"""Unit tests for HoltPredictor -- synchronous methods only (Layer 1).

predict_next() is async and tested in Layer 2 (integration/test_holt_predictor_async.py).
"""

from helios.prediction.holt import HoltPredictor


class TestRecord:
    def test_appends_to_history(self) -> None:
        p = HoltPredictor("m1")
        p.record(5.0)
        p.record(10.0)
        assert list(p._history) == [5.0, 10.0]

    def test_sets_dirty_true(self) -> None:
        p = HoltPredictor("m1")
        p.record(1.0)
        assert p._dirty is True

    def test_updates_peak_when_new_value_exceeds(self) -> None:
        p = HoltPredictor("m1")
        p.record(5.0)
        assert p._peak == 5.0
        p.record(10.0)
        assert p._peak == 10.0

    def test_peak_does_not_decrease(self) -> None:
        p = HoltPredictor("m1")
        p.record(10.0)
        p.record(3.0)
        p.record(1.0)
        assert p._peak == 10.0

    def test_peak_not_updated_for_equal_value(self) -> None:
        p = HoltPredictor("m1")
        p.record(5.0)
        p.record(5.0)
        assert p._peak == 5.0

    def test_history_bounded_by_window_size(self) -> None:
        p = HoltPredictor("m1", window_size=3)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            p.record(v)
        assert list(p._history) == [3.0, 4.0, 5.0]


class TestInitialState:
    def test_dirty_starts_false(self) -> None:
        p = HoltPredictor("m1")
        assert p._dirty is False

    def test_peak_starts_at_one(self) -> None:
        """_peak = 1.0 prevents division by zero in early normalization."""
        p = HoltPredictor("m1")
        assert p._peak == 1.0

    def test_history_starts_empty(self) -> None:
        p = HoltPredictor("m1")
        assert len(p._history) == 0

    def test_fitted_starts_none(self) -> None:
        p = HoltPredictor("m1")
        assert p._fitted is None
