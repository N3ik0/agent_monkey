import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from data.fetcher_router import DataFetcherRouter

@pytest.fixture
def test_router():
    return DataFetcherRouter()

@patch("data.fetcher_router.CCXTDataFetcher")
def test_fetch_auto_routes_to_default_ccxt(mock_ccxt_class, test_router):
    mock_ccxt = mock_ccxt_class.return_value
    mock_df = pd.DataFrame()
    mock_ccxt.fetch.return_value = mock_df
    
    result = test_router.fetch("BTC/USDT")
    
    mock_ccxt.fetch.assert_called_once_with(ticker="BTC/USDT", period="6mo", interval="1d", start=None, end=None)
    assert result is mock_df

@patch("data.fetcher_router.CCXTDataFetcher")
def test_fetch_explicit_provider_kraken(mock_ccxt_class, test_router):
    mock_kraken = MagicMock()
    mock_ccxt_class.return_value = mock_kraken
    
    # Overriding cache manually
    test_router._ccxt_fetchers["kraken"] = mock_kraken
    
    test_router.fetch("kraken:BTC/USD")
    mock_kraken.fetch.assert_called_once_with(ticker="BTC/USD", period="6mo", interval="1d", start=None, end=None)

@patch("data.fetcher_router.CCXTDataFetcher")
def test_fetch_fails_fast_on_error(mock_ccxt_class, test_router):
    mock_ccxt = mock_ccxt_class.return_value
    mock_ccxt.fetch.side_effect = Exception("API Down")
    
    with pytest.raises(RuntimeError, match="Fatal Error: CCXT"):
        test_router.fetch("BTC/USDT")
