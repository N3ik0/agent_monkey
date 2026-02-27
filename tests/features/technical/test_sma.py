import pytest
import pandas as pd
import numpy as np
from features.technical.sma import SMAFeature

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "close": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    })

def test_sma_calculation(sample_df):
    sma = SMAFeature(window=3)
    res = sma.compute(sample_df.copy())
    assert "SMA_3" in res.columns
    # Row index 2 is (10+20+30)/3 = 20
    assert res["SMA_3"].iloc[2] == 20.0
    # First 2 rows should be NaN with a window of 3
    assert pd.isna(res["SMA_3"].iloc[0])
    assert pd.isna(res["SMA_3"].iloc[1])

def test_sma_missing_column():
    sma = SMAFeature(window=3, column="missing_col")
    df = pd.DataFrame({"close": [1, 2, 3]})
    with pytest.raises(KeyError, match="missing from DataFrame"):
        sma.compute(df)

def test_sma_anti_look_ahead(sample_df):
    sma = SMAFeature(window=3)
    
    # Process full dataframe
    res_full = sma.compute(sample_df.copy())
    
    # Process partial dataframe up to index 5
    partial_df = sample_df.iloc[:6].copy()
    res_partial = sma.compute(partial_df)
    
    # Assert values for row 5 are identically the same
    # This proves the feature doesn't look at indices > 5 to calculate index 5
    assert res_full["SMA_3"].iloc[5] == res_partial["SMA_3"].iloc[5]
