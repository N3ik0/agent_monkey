import pandas as pd
from features.base_feature import BaseFeature


class BollingerFeature(BaseFeature):
    """
    Calculates Bollinger Bands for a given price column.
    A volatility indicator that creates an upper and lower envelope
    around a simple moving average.

    Generates three columns: BB_upper_{window}, BB_middle_{window}, BB_lower_{window}.
    """

    def __init__(self, window: int = 20, num_std: float = 2.0, column: str = "close"):
        """
        Initializes the Bollinger Bands feature module.

        Args:
            window (int): The lookback period for the moving average (default 20).
            num_std (float): The number of standard deviations for the bands (default 2.0).
            column (str): The DataFrame column to process (default 'close').
        """
        super().__init__(name=f"BB_{window}")
        self.window = window
        self.num_std = num_std
        self.column = column

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the upper, middle, and lower Bollinger Bands.

        Args:
            df (pd.DataFrame): The market data DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with BB_upper, BB_middle, BB_lower columns.
        """
        if self.column not in df.columns:
            raise KeyError(f"Fatal error: Column '{self.column}' missing from DataFrame.")

        rolling_mean = df[self.column].rolling(window=self.window).mean()
        rolling_std = df[self.column].rolling(window=self.window).std()

        df[f"BB_middle_{self.window}"] = rolling_mean
        df[f"BB_upper_{self.window}"] = rolling_mean + (self.num_std * rolling_std)
        df[f"BB_lower_{self.window}"] = rolling_mean - (self.num_std * rolling_std)

        return df
