"""Tests for MACDFeature."""
import pytest
import pandas as pd
import numpy as np
from features.technical.macd import MACDFeature


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Creates a sample DataFrame with enough rows for MACD calculation."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    return pd.DataFrame({"close": prices})


def test_macd_calculation(sample_df: pd.DataFrame) -> None:
    """Verifies that MACD produces the three expected columns."""
    feature = MACDFeature()
    result = feature.compute(sample_df)

    assert "MACD_line" in result.columns
    assert "MACD_signal" in result.columns
    assert "MACD_histogram" in result.columns
    # Histogram should equal MACD_line - MACD_signal
    pd.testing.assert_series_equal(
        result["MACD_histogram"],
        result["MACD_line"] - result["MACD_signal"],
        check_names=False,
    )


def test_macd_missing_column() -> None:
    """Verifies that MACD raises KeyError when column is missing."""
    df = pd.DataFrame({"open": [1, 2, 3]})
    feature = MACDFeature()
    with pytest.raises(KeyError, match="close"):
        feature.compute(df)


def test_macd_anti_look_ahead(sample_df: pd.DataFrame) -> None:
    """
    Anti-look-ahead-bias test:
    MACD on first 50 rows must match MACD on all 100 rows for row 49.
    """
    feature = MACDFeature()

    partial = feature.compute(sample_df.iloc[:50].copy())
    full = feature.compute(sample_df.copy())

    assert partial["MACD_line"].iloc[-1] == pytest.approx(full["MACD_line"].iloc[49])
    assert partial["MACD_signal"].iloc[-1] == pytest.approx(full["MACD_signal"].iloc[49])


def test_macd_name() -> None:
    """Verifies the feature name for Notion tracking."""
    feature = MACDFeature(fast_period=8, slow_period=21, signal_period=5)
    assert feature.name == "MACD_8_21_5"
