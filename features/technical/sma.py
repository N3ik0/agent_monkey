import pandas as pd
from features.base_feature import BaseFeature

class SMAFeature(BaseFeature):
    """
    Calculates the Simple Moving Average (SMA) for a given price column.
    A foundational trend-following indicator for statistical agents.
    """

    def __init__(self, window: int = 20, column: str = "close"):
        """
        Initializes the SMA feature module.

        Args:
            window (int): The number of periods for the rolling average (e.g., 20, 50, 200).
            column (str): The name of the DataFrame column to process (default is 'close').
        """
        # We pass the dynamic name to the parent class for Notion tracking (e.g., 'SMA_20')
        super().__init__(name=f"SMA_{window}")
        self.window = window
        self.column = column

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the Simple Moving Average and adds it as a new column.

        Args:
            df (pd.DataFrame): The market data DataFrame.

        Returns:
            pd.DataFrame: The mutated DataFrame containing the new SMA column.
        """
        # Fail Fast: Ensure the required column exists before doing math
        if self.column not in df.columns:
            raise KeyError(f"Fatal error: Column '{self.column}' missing from DataFrame.")

        # Calculate the rolling mean
        df[self.name] = df[self.column].rolling(window=self.window).mean()
        
        return df