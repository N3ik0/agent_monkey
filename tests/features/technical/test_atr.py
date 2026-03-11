import pytest
import pandas as pd
import numpy as np
from features.technical.atr import ATRFeature

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "high": [12, 22, 32, 42, 52, 62],
        "low": [8, 18, 28, 38, 48, 58],
        "close": [10, 20, 30, 40, 50, 60]
    })

def test_atr_calculation(sample_df):
    atr = ATRFeature(window=3)
    res = atr.compute(sample_df.copy())
    assert "ATR_3" in res.columns
    # Row 0: TR = 12 - 8 = 4
    # Row 1: TR = max(22-18, |22-10|, |18-10|) = max(4, 12, 8) = 12
    # Row 2: TR = max(32-28, |32-20|, |28-20|) = max(4, 12, 8) = 12
    # ATR_3 for row 2 = (4 + 12 + 12) / 3 = 28 / 3 = 9.333...
    assert res["ATR_3"].iloc[2] == pytest.approx(9.333333333333334)
    assert pd.isna(res["ATR_3"].iloc[0])
    assert pd.isna(res["ATR_3"].iloc[1])

def test_atr_missing_columns():
    atr = ATRFeature(window=3)
    df = pd.DataFrame({"close": [1, 2, 3], "high": [2, 3, 4]})
    with pytest.raises(KeyError, match="missing from DataFrame"):
        atr.compute(df)

def test_atr_anti_look_ahead(sample_df):
    atr = ATRFeature(window=3)
    res_full = atr.compute(sample_df.copy())
    
    partial_df = sample_df.iloc[:4].copy()
    res_partial = atr.compute(partial_df)
    
    assert res_full["ATR_3"].iloc[3] == res_partial["ATR_3"].iloc[3]
