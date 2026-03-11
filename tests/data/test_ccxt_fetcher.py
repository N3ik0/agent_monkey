import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from data.ccxt_fetcher import CCXTDataFetcher

@pytest.fixture
def mock_ccxt_exchange():
    """Mock the ccxt module and its classes."""
    with patch("data.ccxt_fetcher.ccxt") as mock_ccxt:
        mock_exchange = MagicMock()
        mock_ccxt.binance.return_value = mock_exchange
        mock_ccxt.BaseError = Exception
        yield mock_exchange

def test_ccxt_fetcher_init(mock_ccxt_exchange):
    """Test initialization of CCXTDataFetcher."""
    fetcher = CCXTDataFetcher(exchange_id="binance")
    assert fetcher.exchange_id == "binance"
    assert fetcher.exchange == mock_ccxt_exchange

def test_convert_ticker():
    fetcher = CCXTDataFetcher()
    assert fetcher._convert_ticker("BTC/USDT") == "BTC/USDT"
    assert fetcher._convert_ticker("BTC-USD") == "BTC/USDT"
    assert fetcher._convert_ticker("ETH-USD") == "ETH/USDT"

def test_fetch_paginates(mock_ccxt_exchange):
    """Test pagination handles limits gracefully."""
    fetcher = CCXTDataFetcher()
    
    # 3 pages of mock data
    # (chunk[-1][0] + 1) logic requires distinct timestamps
    mock_ccxt_exchange.fetch_ohlcv.side_effect = [
        [[1000, 1, 2, 0.5, 1.5, 10], [2000, 1.5, 2.5, 1, 2, 15]],
        [[3000, 2, 3, 1.5, 2.5, 20], [4000, 2.5, 3.5, 2, 3, 25]],
        []  # Empty chunk to stop
    ]
    
    df = fetcher.fetch("BTC-USD", period="1d")
    
    assert mock_ccxt_exchange.fetch_ohlcv.call_count == 3
    assert len(df) == 4
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "source"]
    assert df.index.name == "Date"
    assert (df["source"] == "binance").all()

def test_fetch_empty_raises(mock_ccxt_exchange):
    """Test empty data raises ValueError."""
    fetcher = CCXTDataFetcher()
    mock_ccxt_exchange.fetch_ohlcv.return_value = []
    
    with pytest.raises(ValueError, match="No data returned"):
        fetcher.fetch("BTC-USD", period="1d")

def test_fetch_with_end_date(mock_ccxt_exchange):
    """Test early break due to end date."""
    fetcher = CCXTDataFetcher()
    
    mock_ccxt_exchange.fetch_ohlcv.side_effect = [
        [[1000, 1, 2, 0.5, 1.5, 10]],
        [[1704067200000, 1.5, 2.5, 1, 2, 15]] # 2024-01-01
    ]
    
    # It should break pagination on exact match or beyond
    with patch.object(fetcher, '_validate', return_value=pd.DataFrame()):
        # Mock validate to avoid column checks since we return a mock df during test debugging if needed
        pass
    
    # We will just see if it runs without crashing
    df = fetcher.fetch("BTC-USD", start="2023-01-01", end="2023-12-31")
    assert len(df) == 1 # Second one is filtered out
