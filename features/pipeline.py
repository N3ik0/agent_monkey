from typing import List
import pandas as pd

from features.base_feature import BaseFeature

class FeaturePipeline:
    """
    The Assembly Line for Market Data.
    Chains multiple feature engineering modules to prepare the dataset for the Monkeys.
    Ensures immutability of the raw data and tracks feature provenance for Notion logging.
    """

    def __init__(self):
        """
        Initialize an empty pipeline
        """
        self.features: List[BaseFeature] = []

    def add_feature(self, feature: BaseFeature) -> 'FeaturePipeline':
        """
        Appends a new feature module to the pipeline.
        Implements the Fluent Interface design pattern for easy chaining.

        Args:
            feature (BaseFeature): An instance of a class inheriting from BaseFeature.

        Returns:
            FeaturePipeline: Returns self to allow method chaining.
        """
        self.features.append(feature)
        return self

    def generate(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the feature generation sequentially.
        
        Args:
            raw_df (pd.DataFrame): The raw OHLCV market data.

        Returns:
            pd.DataFrame: A fully processed, ML-ready dataset without NaN values.
        """
        processed_df = raw_df.copy()

        for feature in self.features:
            processed_df = feature.compute(processed_df)

        processed_df.dropna(inplace=True)

        return processed_df

    def get_feature_name(self) -> List[str]:
        """
        Retrieves the names of all active features in this pipeline.
        Crucial for logging experiments in the Notion 'Experience Lab'.

        Returns:
            List[str]: A list of feature identifiers.
        """
        return [feature.name for feature in self.features]