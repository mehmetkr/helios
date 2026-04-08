"""HoltPredictor -- per-model demand predictor using Holt's linear trend method.

Double exponential smoothing (trend only, no seasonal component).
ExponentialSmoothing(trend="add", seasonal=None) is Holt's method, not ETS.
Holt's method captures trend with shorter history requirements than ETS
(min ~10 observations vs two full seasonal periods).

Produces a normalized demand score in [0.0, 1.0] relative to historical peak,
for dimensionally-consistent use in CostBasedEvictionPolicy.
"""

import asyncio
from collections import deque
from typing import Any

from statsmodels.tsa.holtwinters import ExponentialSmoothing


class HoltPredictor:
    """Per-model demand predictor with lazy dirty-flag refit."""

    MIN_HISTORY = 10

    def __init__(self, model_id: str, window_size: int = 60) -> None:
        self.model_id = model_id
        self._history: deque[float] = deque(maxlen=window_size)  # bounded
        self._peak: float = 1.0
        self._fitted: Any = None
        self._dirty: bool = False
        self._lock = asyncio.Lock()  # prevents duplicate refits under concurrency

    def record(self, request_count: float) -> None:
        """Record a new observation. Marks predictor dirty for lazy refit."""
        self._history.append(request_count)
        if request_count > self._peak:
            self._peak = request_count
        self._dirty = True

    async def predict_next(self) -> float:
        """Return normalized demand forecast in [0.0, 1.0].

        Refits only when new data has arrived (dirty flag).
        fit() runs in asyncio.to_thread() -- never blocks the event loop.
        Lock prevents duplicate refits from concurrent callers.
        """
        if len(self._history) < self.MIN_HISTORY:
            return 0.0  # warm-up period: no forecast, no pre-warming

        async with self._lock:
            if self._dirty or self._fitted is None:
                snapshot = list(self._history)
                self._fitted = await asyncio.to_thread(
                    lambda: ExponentialSmoothing(snapshot, trend="add", seasonal=None).fit(
                        optimized=True
                    )
                )
                self._dirty = False

        raw = float(self._fitted.forecast(1)[0])
        return min(max(raw / self._peak, 0.0), 1.0)
