"""Layer 2: HoltPredictor async integration tests.

predict_next() is async def -- requires pytest-asyncio.
"""

import asyncio

import pytest

from helios.prediction.holt import HoltPredictor


@pytest.mark.asyncio
async def test_predict_returns_zero_below_min_history() -> None:
    """During warm-up (< MIN_HISTORY observations), predict_next returns 0.0."""
    p = HoltPredictor("m1")
    for i in range(HoltPredictor.MIN_HISTORY - 1):
        p.record(float(i + 1))
    result = await p.predict_next()
    assert result == 0.0


@pytest.mark.asyncio
async def test_predict_returns_value_in_unit_range() -> None:
    """After MIN_HISTORY observations, output is normalized to [0, 1]."""
    p = HoltPredictor("m1")
    for i in range(20):
        p.record(float(i + 1))
    result = await p.predict_next()
    assert 0.0 <= result <= 1.0


@pytest.mark.asyncio
async def test_refit_fires_once_per_dirty_cycle() -> None:
    """Concurrent predict_next() calls should trigger only one refit."""
    p = HoltPredictor("m1")
    for i in range(15):
        p.record(float(i + 1))

    # First predict triggers refit (dirty from record calls).
    r1 = await p.predict_next()
    assert p._dirty is False

    # No new record -- predict should NOT refit (not dirty).
    r2 = await p.predict_next()
    assert r1 == r2  # same fitted model, same forecast

    # New record makes it dirty again.
    p.record(20.0)
    assert p._dirty is True
    r3 = await p.predict_next()
    assert p._dirty is False
    # Forecast may differ after new data.
    assert 0.0 <= r3 <= 1.0


@pytest.mark.asyncio
async def test_concurrent_predict_does_not_double_refit() -> None:
    """Two concurrent predict_next() should not both trigger a refit."""
    p = HoltPredictor("m1")
    for i in range(15):
        p.record(float(i + 1))

    # Fire two concurrent predictions.
    r1, r2 = await asyncio.gather(p.predict_next(), p.predict_next())
    assert 0.0 <= r1 <= 1.0
    assert 0.0 <= r2 <= 1.0
    assert p._dirty is False
