import pytest
import pandas as pd
import numpy as np
from features.technical.rsi import RSIFeature

@pytest.fixture
def sample_df():
    # A mix of ups and downs to test RSI
    return pd.DataFrame({
        "close": [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28]
    })

def test_rsi_calculation(sample_df):
    rsi = RSIFeature(window=14)
    res = rsi.compute(sample_df.copy())
    assert "RSI_14" in res.columns
    # Row 14 is the first one with a valid rolling 14-period average
    assert not pd.isna(res["RSI_14"].iloc[14])

def test_rsi_missing_column():
    rsi = RSIFeature(window=14, column="missing_col")
    df = pd.DataFrame({"close": [1, 2, 3]})
    with pytest.raises(KeyError, match="missing from DataFrame"):
        rsi.compute(df)

def test_rsi_anti_look_ahead(sample_df):
    rsi = RSIFeature(window=10)
    
    # Process full dataframe
    res_full = rsi.compute(sample_df.copy())
    
    # Process partial dataframe up to index 11
    partial_df = sample_df.iloc[:12].copy()
    res_partial = rsi.compute(partial_df)
    
    # Assert values for row 11 are identically the same
    assert res_full["RSI_10"].iloc[11] == res_partial["RSI_10"].iloc[11]
