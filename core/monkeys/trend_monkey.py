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
        if self.fast_col not in market_data.columns or self.slow_col not in market_data.columns or "close" not in market_data.columns:
            # According to rules, we must explicitly crash if data is corrupted/missing
            raise KeyError(f"Fatal error: Missing columns for {self.name}. Required: {self.fast_col}, {self.slow_col}, close")

        # Need at least two rows to check a crossover if we wanted to be precise,
        # but for a simple state-based signal, we can just check the latest row.
        latest = market_data.iloc[-1]
        
        # If NaN values are present, wait
        if pd.isna(latest[self.fast_col]) or pd.isna(latest[self.slow_col]) or pd.isna(latest["close"]):
            return MonkeySignal(self.name, Action.WAIT, 0.0)

        fast_val = float(latest[self.fast_col])
        slow_val = float(latest[self.slow_col])
        close_val = float(latest["close"])
        
        # Calculate percentage difference between fast and slow
        diff = abs(fast_val - slow_val) / slow_val
        
        # Momentum Filter: Do not act in tight consolidation ranges (< 0.3% spread)
        if diff <= 0.003:
            return MonkeySignal(self.name, Action.WAIT, 0.0)
            
        # Confidence can be scaled by the percentage difference, capped at 1.0
        # A 5% difference gives 100% confidence
        confidence = min(diff * 20.0, 1.0)

        if fast_val > slow_val:
            # Price Confirmation: Do not buy if price is below the slow trend
            if close_val <= slow_val:
                return MonkeySignal(self.name, Action.WAIT, 0.0)
            return MonkeySignal(self.name, Action.BUY, confidence)
            
        elif fast_val < slow_val:
            # Price Confirmation: Do not sell if price is above the slow trend
            if close_val >= slow_val:
                return MonkeySignal(self.name, Action.WAIT, 0.0)
            return MonkeySignal(self.name, Action.SELL, confidence)
            
        else:
            return MonkeySignal(self.name, Action.WAIT, 0.0)
