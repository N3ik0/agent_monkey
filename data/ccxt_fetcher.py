from typing import Optional
from datetime import datetime, timedelta, timezone
import pandas as pd
import ccxt

from data.base_fetcher import BaseDataFetcher


class CCXTDataFetcher(BaseDataFetcher):
    """
    Concrete data fetcher using the CCXT library.
    Supports 100+ crypto exchanges with a unified interface.
    Defaults to Binance for public OHLCV data (no API key required).
    """

    # Mapping yfinance style periods → Python timedelta
    # Used by _period_to_since() to calculate the start timestamp
    PERIOD_MAP = {
        "1d":  timedelta(days=1),
        "5d":  timedelta(days=5),
        "1mo": timedelta(days=30),
        "3mo": timedelta(days=90),
        "6mo": timedelta(days=180),
        "1y":  timedelta(days=365),
        "2y":  timedelta(days=730),
    }

    # Mapping yfinance intervals → CCXT format
    # yfinance uses "1wk", CCXT uses "1w" for example
    INTERVAL_MAP = {
        "1m":  "1m",
        "5m":  "5m",
        "15m": "15m",
        "30m": "30m",
        "1h":  "1h",
        "4h":  "4h",
        "1d":  "1d",
        "1wk": "1w",
        "1mo": "1M",
    }

    def __init__(
        self,
        exchange_id: str = "binance",
        api_key: Optional[str] = None,
        secret: Optional[str] = None,
    ):
        """
        Initializes the CCXT exchange.

        Args:
            exchange_id (str): Name of the CCXT exchange ('binance', 'bybit', 'kraken'...).
                               Default is Binance — public OHLCV access without API key.
            api_key (Optional[str]): API key (optional for public read).
            secret (Optional[str]): API secret (optional for public read).
        """
        # getattr(ccxt, "binance") is equivalent to ccxt.binance()
        # This allows choosing the exchange dynamically via a string
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange: ccxt.Exchange = exchange_class({
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,  # Automatically respects API rate limits
        })
        self.exchange_id = exchange_id

    def _convert_ticker(self, ticker: str) -> str:
        """
        Converts a yfinance ticker to a CCXT symbol.

        yfinance → CCXT :
            "BTC-USD"  → "BTC/USDT"
            "ETH-USD"  → "ETH/USDT"
            "BTC/USDT" → "BTC/USDT"  (already correct format, passthrough)

        We replace USD with USDT because the majority of liquid pairs
        on centralized exchanges are quoted in USDT, not USD.

        Args:
            ticker (str): Symbol in yfinance or CCXT format.

        Returns:
            str: Symbol in CCXT format (e.g., "BTC/USDT").
        """
        if "/" in ticker:
            return ticker  # Already in CCXT format
        # "BTC-USD" → split on "-" → ["BTC", "USD"] → "BTC/USDT"
        parts = ticker.replace("-USD", "-USDT").split("-")
        return "/".join(parts)

    def _period_to_since(self, period: str) -> Optional[int]:
        """
        Converts a yfinance style period to a Unix timestamp (milliseconds).

        CCXT expects a Unix timestamp in ms for the 'since' parameter.
        Ex: "6mo" → timestamp of 180 days ago in milliseconds.

        Args:
            period (str): yfinance style period ('1d', '6mo', '1y'...).

        Returns:
            Optional[int]: Unix timestamp in milliseconds, or None if unknown period.
        """
        delta = self.PERIOD_MAP.get(period)
        if delta is None:
            return None  # If unknown period, CCXT will return the most recent data
        since_dt = datetime.now(timezone.utc) - delta
        # Python datetime.timestamp() returns seconds → *1000 for milliseconds
        return int(since_dt.timestamp() * 1000)

    def _interval_to_ccxt(self, interval: str) -> str:
        """
        Converts a yfinance interval to a CCXT timeframe.

        Args:
            interval (str): yfinance interval ('1d', '1h', '1wk'...).

        Returns:
            str: CCXT timeframe ('1d', '1h', '1w'...).
                 If not found in the mapping, returns the interval as is.
        """
        return self.INTERVAL_MAP.get(interval, interval)

    def fetch(
        self,
        ticker: str,
        period: str = "6mo",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches OHLCV data via CCXT for a crypto asset.

        CCXT returns a list of lists in the format:
            [[timestamp_ms, open, high, low, close, volume], ...]
        We reconstruct a clean pandas DataFrame with a DatetimeIndex.

        Args:
            ticker (str): Symbol (yfinance format 'BTC-USD' or CCXT 'BTC/USDT').
            period (str): Lookback period ('6mo', '1y'...). Ignored if start/end are provided.
            interval (str): Candle interval ('1d', '1h'...).
            start (Optional[str]): ISO start date ('2024-01-01'). Takes precedence over period.
            end (Optional[str]): ISO end date ('2024-12-31'). Not used by CCXT
                                 (we filter manually after fetch).

        Returns:
            pd.DataFrame: OHLCV DataFrame with lowercase columns and UTC DatetimeIndex.

        Raises:
            ValueError: If no data is returned or columns are missing.
            ccxt.BaseError: If the exchange returns an error (unknown symbol, etc.).
        """
        ccxt_symbol = self._convert_ticker(ticker)
        ccxt_timeframe = self._interval_to_ccxt(interval)

        # Calculate start timestamp
        if start:
            since = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        else:
            since = self._period_to_since(period)

        raw_data = []
        current_since = since
        
        # Pagination loop to fetch all historical data needed (bypassing exchange limits e.g. 1000 candles)
        while True:
            chunk = self.exchange.fetch_ohlcv(
                symbol=ccxt_symbol,
                timeframe=ccxt_timeframe,
                since=current_since,
                limit=1000,
            )

            if not chunk:
                break

            # Prevent infinite loop if API returns the same data
            if raw_data and chunk[-1][0] <= raw_data[-1][0]:
                break

            raw_data.extend(chunk)
            current_since = chunk[-1][0] + 1  # +1ms to avoid fetching the exact same last candle

            # Early break if we have reached the end date
            if end:
                end_ts = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
                if current_since > end_ts:
                    break

        if not raw_data:
            raise ValueError(
                f"No data returned by {self.exchange_id} "
                f"for the symbol '{ccxt_symbol}' ({ccxt_timeframe})."
            )

        # Reconstruct DataFrame from CCXT list of lists
        df = pd.DataFrame(raw_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Convert Unix ms timestamp to UTC DatetimeIndex
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df.index.name = "Date"

        # Optional filter on end date
        if end:
            end_dt = pd.Timestamp(end, tz="UTC")
            df = df[df.index <= end_dt]

        # Inject the source of the data
        df["source"] = self.exchange_id

        return self._validate(df, ticker)
