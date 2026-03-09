"""Tests for BollingerFeature."""
import pytest
import pandas as pd
import numpy as np
from features.technical.bollinger import BollingerFeature


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Creates a sample DataFrame with enough rows for Bollinger calculation."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    return pd.DataFrame({"close": prices})


def test_bollinger_calculation(sample_df: pd.DataFrame) -> None:
    """Verifies that Bollinger Bands produces the three expected columns."""
    feature = BollingerFeature(window=20)
    result = feature.compute(sample_df)

    assert "BB_upper_20" in result.columns
    assert "BB_middle_20" in result.columns
    assert "BB_lower_20" in result.columns

    # Upper should always be >= middle >= lower (where not NaN)
    valid = result.dropna(subset=["BB_upper_20", "BB_middle_20", "BB_lower_20"])
    assert (valid["BB_upper_20"] >= valid["BB_middle_20"]).all()
    assert (valid["BB_middle_20"] >= valid["BB_lower_20"]).all()


def test_bollinger_missing_column() -> None:
    """Verifies that Bollinger raises KeyError when column is missing."""
    df = pd.DataFrame({"open": [1, 2, 3]})
    feature = BollingerFeature()
    with pytest.raises(KeyError, match="close"):
        feature.compute(df)


def test_bollinger_anti_look_ahead(sample_df: pd.DataFrame) -> None:
    """
    Anti-look-ahead-bias test:
    Bollinger on first 50 rows must match Bollinger on all 100 rows for row 49.
    """
    feature = BollingerFeature(window=20)

    partial = feature.compute(sample_df.iloc[:50].copy())
    full = feature.compute(sample_df.copy())

    assert partial["BB_middle_20"].iloc[-1] == pytest.approx(full["BB_middle_20"].iloc[49])
    assert partial["BB_upper_20"].iloc[-1] == pytest.approx(full["BB_upper_20"].iloc[49])
    assert partial["BB_lower_20"].iloc[-1] == pytest.approx(full["BB_lower_20"].iloc[49])


def test_bollinger_symmetry(sample_df: pd.DataFrame) -> None:
    """Upper and lower bands should be symmetric around the middle."""
    feature = BollingerFeature(window=20, num_std=2.0)
    result = feature.compute(sample_df)
    valid = result.dropna(subset=["BB_upper_20", "BB_middle_20", "BB_lower_20"])

    upper_diff = valid["BB_upper_20"] - valid["BB_middle_20"]
    lower_diff = valid["BB_middle_20"] - valid["BB_lower_20"]
    pd.testing.assert_series_equal(upper_diff, lower_diff, check_names=False)
