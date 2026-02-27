import pandas as pd
from features.base_feature import BaseFeature

class RSIFeature(BaseFeature):
    """
    Calculates the Relative Strength Index (RSI) for a given price column.
    A momentum oscillator that measures the speed and change of price movements.
    """

    def __init__(self, window: int = 14, column: str = "close"):
        """
        Initializes the RSI feature module.

        Args:
            window (int): The lookback period for RSI calculation.
            column (str): The name of the DataFrame column to process (default is 'close').
        """
        super().__init__(name=f"RSI_{window}")
        self.window = window
        self.column = column

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the Relative Strength Index.

        Args:
            df (pd.DataFrame): The market data DataFrame.

        Returns:
            pd.DataFrame: The mutated DataFrame containing the new RSI column.
        """
        if self.column not in df.columns:
            raise KeyError(f"Fatal error: Column '{self.column}' missing from DataFrame.")

        delta = df[self.column].diff()
        
        # Ensure we don't look ahead by calculating positive and negative gains
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean()
        
        rs = gain / loss
        # Guard against zero division
        df[self.name] = 100 - (100 / (1 + rs))
        df[self.name] = df[self.name].fillna(100.0).where(loss == 0, df[self.name]) # If loss is 0, RSI is 100
        
        return df
