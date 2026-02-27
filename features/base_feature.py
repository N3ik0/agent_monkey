from abc import ABC, abstractmethod
import pandas as pd

class BaseFeature(ABC):
    """
    Abstract Base Class for all feature engineering modules.
    Ensures a standardized way to compute technical indicators, statistical 
    transformations, or computer vision matrix formulations.
    """

    def __init__(self, name: str):
        """
        Initialize the feature module.
        Args : 
            name (str): The unique identifier for this feature (e.g: 'RSI_14)
                        Used for tracking the notion
        """
        self.name = name

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the feature and appends it to the market data.
        
        WARNING (Quant Rule): Do NOT introduce Look-ahead bias here. 
        Never use future data (e.g., df['close'].shift(-1)) to compute a current feature.
        
        Args:
            df (pd.DataFrame): The raw or partially processed OHLCV market data.
            
        Returns:
            pd.DataFrame: A new DataFrame containing the original columns plus the computed feature.
        """
        pass