"""Tests for YFinanceDataFetcher (mocked — no live API calls)."""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from data.yfinance_fetcher import YFinanceDataFetcher


@pytest.fixture
def mock_ohlcv_df() -> pd.DataFrame:
    """Creates a sample OHLCV DataFrame mimicking yfinance output."""
    return pd.DataFrame({
        "Open": [100.0, 101.0, 102.0],
        "High": [105.0, 106.0, 107.0],
        "Low": [99.0, 100.0, 101.0],
        "Close": [104.0, 105.0, 106.0],
        "Volume": [1000, 1100, 1200],
        "Dividends": [0.0, 0.0, 0.0],
        "Stock Splits": [0, 0, 0],
    })


@patch("data.yfinance_fetcher.yf")
def test_fetch_normalizes_columns(mock_yf: MagicMock, mock_ohlcv_df: pd.DataFrame) -> None:
    """Verifies that column names are normalized to lowercase."""
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_ohlcv_df
    mock_yf.Ticker.return_value = mock_ticker

    fetcher = YFinanceDataFetcher()
    result = fetcher.fetch(ticker="AAPL", period="6mo")

    assert "close" in result.columns
    assert "open" in result.columns
    assert "high" in result.columns
    assert "low" in result.columns
    assert "volume" in result.columns
    # Non-OHLCV columns should be dropped
    assert "dividends" not in result.columns
    assert "stock splits" not in result.columns


@patch("data.yfinance_fetcher.yf")
def test_fetch_empty_data_raises(mock_yf: MagicMock) -> None:
    """Verifies that an empty DataFrame raises ValueError."""
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    mock_yf.Ticker.return_value = mock_ticker

    fetcher = YFinanceDataFetcher()
    with pytest.raises(ValueError, match="No data returned"):
        fetcher.fetch(ticker="INVALID_TICKER")


@patch("data.yfinance_fetcher.yf")
def test_fetch_uses_start_end(mock_yf: MagicMock, mock_ohlcv_df: pd.DataFrame) -> None:
    """Verifies that start/end parameters are passed to yfinance."""
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_ohlcv_df
    mock_yf.Ticker.return_value = mock_ticker

    fetcher = YFinanceDataFetcher()
    fetcher.fetch(ticker="AAPL", start="2024-01-01", end="2024-12-31")

    mock_ticker.history.assert_called_once_with(
        start="2024-01-01", end="2024-12-31", interval="1d"
    )
