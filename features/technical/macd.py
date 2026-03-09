import pandas as pd
from features.base_feature import BaseFeature


class MACDFeature(BaseFeature):
    """
    Calculates the Moving Average Convergence Divergence (MACD).
    A trend-following momentum indicator that shows the relationship
    between two exponential moving averages.

    Generates three columns: MACD_line, MACD_signal, MACD_histogram.
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = "close",
    ):
        """
        Initializes the MACD feature module.

        Args:
            fast_period (int): Period for the fast EMA (default 12).
            slow_period (int): Period for the slow EMA (default 26).
            signal_period (int): Period for the signal line EMA (default 9).
            column (str): The DataFrame column to process (default 'close').
        """
        super().__init__(name=f"MACD_{fast_period}_{slow_period}_{signal_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.column = column

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes MACD line, signal line, and histogram.

        Args:
            df (pd.DataFrame): The market data DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with MACD_line, MACD_signal, MACD_histogram columns.
        """
        if self.column not in df.columns:
            raise KeyError(f"Fatal error: Column '{self.column}' missing from DataFrame.")

        # MACD Line = Fast EMA - Slow EMA
        fast_ema = df[self.column].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = df[self.column].ewm(span=self.slow_period, adjust=False).mean()
        df["MACD_line"] = fast_ema - slow_ema

        # Signal Line = EMA of the MACD Line
        df["MACD_signal"] = df["MACD_line"].ewm(span=self.signal_period, adjust=False).mean()

        # Histogram = MACD Line - Signal Line
        df["MACD_histogram"] = df["MACD_line"] - df["MACD_signal"]

        return df
