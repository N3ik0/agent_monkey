import pandas as pd
from features.base_feature import BaseFeature

class EMAFeature(BaseFeature):
    """
    Calculates the Exponential Moving Average (EMA) for a given price column.
    A trend-following indicator that gives more weight to recent prices.
    """

    def __init__(self, window: int = 20, column: str = "close"):
        """
        Initializes the EMA feature module.

        Args:
            window (int): The number of periods for the exponential average (e.g., 20, 50, 200).
            column (str): The name of the DataFrame column to process (default is 'close').
        """
        super().__init__(name=f"EMA_{window}")
        self.window = window
        self.column = column

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the Exponential Moving Average and adds it as a new column.

        Args:
            df (pd.DataFrame): The market data DataFrame.

        Returns:
            pd.DataFrame: The mutated DataFrame containing the new EMA column.
        """
        # Fail Fast: Ensure the required column exists before doing math
        if self.column not in df.columns:
            raise KeyError(f"Fatal error: Column '{self.column}' missing from DataFrame.")

        # Calculate the exponential moving average
        df[self.name] = df[self.column].ewm(span=self.window, adjust=False).mean()
        
        return df
