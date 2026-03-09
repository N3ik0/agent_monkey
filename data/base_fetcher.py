from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class BaseDataFetcher(ABC):
    """
    Abstract Base Class for all data acquisition modules.
    Ensures a standardized interface for fetching OHLCV market data,
    regardless of the underlying data source (API, CSV, WebSocket).
    """

    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

    @abstractmethod
    def fetch(
        self,
        ticker: str,
        period: str = "6mo",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches OHLCV market data for the specified ticker.

        Args:
            ticker (str): The asset symbol (e.g., 'BTC-USD', 'AAPL').
            period (str): The lookback period (e.g., '6mo', '1y'). Ignored if start/end are set.
            interval (str): The candle interval (e.g., '1d', '1h').
            start (Optional[str]): Start date string (e.g., '2024-01-01').
            end (Optional[str]): End date string (e.g., '2024-12-31').

        Returns:
            pd.DataFrame: A DataFrame with lowercase columns: open, high, low, close, volume.

        Raises:
            ValueError: If the fetched data is empty or missing required columns.
        """
        pass

    def _validate(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Validates that the fetched DataFrame meets the minimum requirements.

        Args:
            df (pd.DataFrame): The raw fetched data.
            ticker (str): The ticker symbol, used for error messages.

        Returns:
            pd.DataFrame: The validated DataFrame.

        Raises:
            ValueError: If the DataFrame is empty or missing required OHLCV columns.
        """
        if df.empty:
            raise ValueError(
                f"Fatal error: No data returned for ticker '{ticker}'. "
                f"Check the ticker symbol and date range."
            )

        missing_cols = [
            col for col in self.REQUIRED_COLUMNS if col not in df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Fatal error: Missing required columns {missing_cols} "
                f"for ticker '{ticker}'. Available: {list(df.columns)}"
            )

        return df
