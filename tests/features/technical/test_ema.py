import pytest
import pandas as pd
from features.technical.ema import EMAFeature

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "close": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    })

def test_ema_calculation(sample_df):
    ema = EMAFeature(window=3)
    res = ema.compute(sample_df.copy())
    assert "EMA_3" in res.columns
    # EWM with adjust=False starts matching the first value
    assert res["EMA_3"].iloc[0] == 10.0
    # Next value is (20 - 10) * (2/4) + 10 = 15.0
    assert res["EMA_3"].iloc[1] == 15.0

def test_ema_missing_column():
    ema = EMAFeature(window=3, column="missing_col")
    df = pd.DataFrame({"close": [1, 2, 3]})
    with pytest.raises(KeyError, match="missing from DataFrame"):
        ema.compute(df)

def test_ema_anti_look_ahead(sample_df):
    ema = EMAFeature(window=3)
    
    # Process full dataframe
    res_full = ema.compute(sample_df.copy())
    
    # Process partial dataframe up to index 5
    partial_df = sample_df.iloc[:6].copy()
    res_partial = ema.compute(partial_df)
    
    # Assert values for row 5 are identically the same
    assert res_full["EMA_3"].iloc[5] == res_partial["EMA_3"].iloc[5]
