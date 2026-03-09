from typing import Optional
import pandas as pd
import yfinance as yf

from data.base_fetcher import BaseDataFetcher


class YFinanceDataFetcher(BaseDataFetcher):
    """
    Concrete data fetcher using the yfinance library.
    Downloads OHLCV data from Yahoo Finance and normalizes column names.
    """

    def fetch(
        self,
        ticker: str,
        period: str = "6mo",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches historical OHLCV data from Yahoo Finance.

        Args:
            ticker (str): The asset symbol (e.g., 'BTC-USD', 'AAPL').
            period (str): The lookback period (e.g., '6mo', '1y'). Ignored if start/end are set.
            interval (str): The candle interval (e.g., '1d', '1h').
            start (Optional[str]): Start date string (e.g., '2024-01-01').
            end (Optional[str]): End date string (e.g., '2024-12-31').

        Returns:
            pd.DataFrame: A cleaned DataFrame with lowercase OHLCV columns.

        Raises:
            ValueError: If no data is returned or required columns are missing.
        """
        asset = yf.Ticker(ticker)

        if start and end:
            raw_df = asset.history(start=start, end=end, interval=interval)
        else:
            raw_df = asset.history(period=period, interval=interval)

        # Normalize column names to lowercase for consistency
        raw_df.columns = [col.lower() for col in raw_df.columns]

        # Drop non-OHLCV columns (e.g., 'dividends', 'stock splits')
        cols_to_keep = [
            col for col in raw_df.columns
            if col in self.REQUIRED_COLUMNS
        ]
        clean_df = raw_df[cols_to_keep].copy()

        # Validate the result
        return self._validate(clean_df, ticker)
