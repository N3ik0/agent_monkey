import numpy as np
import pandas as pd
from features.base_feature import BaseFeature

class ATRFeature(BaseFeature):
    """
    Calculates the Average True Range (ATR).
    Measures market volatility.
    """

    def __init__(self, window: int = 14):
        """
        Initializes the ATR feature module.
        
        Args:
            window (int): The number of periods for the rolling average.
        """
        super().__init__(name=f"ATR_{window}")
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the ATR and adds it as a new column.

        Args:
            df (pd.DataFrame): The market data DataFrame.

        Returns:
            pd.DataFrame: The mutated DataFrame containing the new ATR column.
        """
        required_cols = ["high", "low", "close"]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Fatal error: Column '{col}' missing from DataFrame.")

        # Calculate True Range (TR)
        # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        
        # We use pandas vectorized operations for performance and avoiding look-ahead bias
        high_low = df["high"] - df["low"]
        high_prev_close = (df["high"] - df["close"].shift(1)).abs()
        low_prev_close = (df["low"] - df["close"].shift(1)).abs()
        
        # TR is the element-wise maximum of these three arrays
        tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
        
        df[self.name] = tr.rolling(window=self.window).mean()
        
        return df
