from core.types import Action, MonkeySignal
from core.base_monkey import BaseMonkey
import pandas as pd

class TrendMonkey(BaseMonkey):
    """
    A simple trend-following monkey that buys when the fast moving average
    crosses above the slow moving average, and sells when it crosses below.
    """

    def __init__(self, name: str, fast_col: str = "SMA_20", slow_col: str = "SMA_50", weight: float = 1.0):
        super().__init__(name, weight)
        self.fast_col = fast_col
        self.slow_col = slow_col

    def analyze(self, market_data: pd.DataFrame) -> MonkeySignal:
        if self.fast_col not in market_data.columns or self.slow_col not in market_data.columns:
            # According to rules, we must explicitly crash if data is corrupted/missing
            raise KeyError(f"Fatal error: Missing columns for {self.name}. Required: {self.fast_col}, {self.slow_col}")

        # Need at least two rows to check a crossover if we wanted to be precise,
        # but for a simple state-based signal, we can just check the latest row.
        latest = market_data.iloc[-1]
        
        # If NaN values are present, wait
        if pd.isna(latest[self.fast_col]) or pd.isna(latest[self.slow_col]):
            return MonkeySignal(self.name, Action.WAIT, 0.0)

        # Simple logic: Fast > Slow -> trend is up (BUY)
        # Fast < Slow -> trend is down (SELL)
        # Fast == Slow -> WAIT
        
        fast_val = latest[self.fast_col]
        slow_val = latest[self.slow_col]
        
        # Confidence can be scaled by the percentage difference, capped at 1.0
        diff = abs(fast_val - slow_val) / slow_val
        # Let's say a 5% difference gives 100% confidence
        confidence = min(diff * 20.0, 1.0)

        if fast_val > slow_val:
            return MonkeySignal(self.name, Action.BUY, confidence)
        elif fast_val < slow_val:
            return MonkeySignal(self.name, Action.SELL, confidence)
        else:
            return MonkeySignal(self.name, Action.WAIT, 0.0)
