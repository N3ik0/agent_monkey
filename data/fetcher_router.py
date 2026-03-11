import warnings
from typing import Optional
import pandas as pd

from data.base_fetcher import BaseDataFetcher
from data.ccxt_fetcher import CCXTDataFetcher


class DataFetcherRouter(BaseDataFetcher):
    """
    Intelligent data source router.

    Automatically routes the request to a CCXT exchange.
    - Default is Binance for Crypto.
    - Explicit routing can be done via prefix: 'kraken:BTC/USDT'

    This is the ONLY entry point to be used in the rest of the code.
    No other module should instantiate a specific fetcher directly.
    """

    def __init__(
        self,
        default_crypto_exchange: str = "binance",
        api_keys: Optional[dict] = None,
    ):
        """
        Initializes the router with the CCXT exchanges.

        Args:
            default_crypto_exchange (str): Default CCXT exchange for cryptos (default: 'binance').
            api_keys (Optional[dict]): Dictionary mapping exchange names to their API keys.
                                       e.g., {"binance": {"api_key": "x", "secret": "y"}}
        """
        self.default_crypto_exchange = default_crypto_exchange
        self.api_keys = api_keys or {}
        
        # Cache for dynamically instantiated CCXT fetchers
        self._ccxt_fetchers = {}
        
    def _get_ccxt_fetcher(self, exchange_id: str) -> 'CCXTDataFetcher':
        """Retrieves or creates a CCXT fetcher for a given exchange."""
        if exchange_id not in self._ccxt_fetchers:
            keys = self.api_keys.get(exchange_id, {})
            self._ccxt_fetchers[exchange_id] = CCXTDataFetcher(
                exchange_id=exchange_id,
                api_key=keys.get("api_key"),
                secret=keys.get("secret"),
            )
        return self._ccxt_fetchers[exchange_id]

    def fetch(
        self,
        ticker: str,
        period: str = "6mo",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Universal entry point: routes to the correct source based on the asset.

        Supports explicit provider routing via prefix:
        e.g., "kraken:BTC/USDT" forces CCXT via Kraken.
        If no prefix, automatically uses the default CCXT exchange.

        Args:
            ticker (str): Asset symbol, optionally prefixed with provider (e.g., 'binance:BTC/USDT').
            period (str): Lookback period.
            interval (str): Candle interval.
            start (Optional[str]): Start date.
            end (Optional[str]): End date.

        Returns:
            pd.DataFrame: Normalized OHLCV DataFrame, transparent source for the caller.
        """
        # Parse explicit provider prefix from ticker (e.g. "kraken:BTC/USDT")
        exchange_id = self.default_crypto_exchange
        clean_ticker = ticker
        
        if ":" in ticker:
            prefix, clean_ticker = ticker.split(":", 1)
            exchange_id = prefix.lower()
        
        kwargs = dict(ticker=clean_ticker, period=period, interval=interval, start=start, end=end)

        try:
            fetcher = self._get_ccxt_fetcher(exchange_id)
            df = fetcher.fetch(**kwargs)
            print(f"✅ [{clean_ticker}] Data fetched via CCXT ({exchange_id})")
            return df
        except Exception as e:
            # We fail fast, no silent fallback to avoid dissonances or bad data integrations
            raise RuntimeError(
                f"🚨 Fatal Error: CCXT ({exchange_id}) failed to fetch data for '{clean_ticker}'. "
                f"Original error: {e}"
            ) from e
