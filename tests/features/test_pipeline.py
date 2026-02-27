import pytest
import pandas as pd
from features.pipeline import FeaturePipeline
from features.base_feature import BaseFeature

class DummyFeature(BaseFeature):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.name] = 1
        return df

class DropAllFeature(BaseFeature):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.name] = [pd.NA if i % 2 == 0 else 1 for i in range(len(df))]
        return df

def test_pipeline_chaining():
    pipeline = FeaturePipeline()
    f1 = DummyFeature("F1")
    f2 = DummyFeature("F2")
    pipeline.add_feature(f1).add_feature(f2)
    assert pipeline.get_feature_name() == ["F1", "F2"]

def test_pipeline_generate():
    pipeline = FeaturePipeline()
    pipeline.add_feature(DummyFeature("F1"))
    df = pd.DataFrame({"close": [1, 2, 3]})
    res = pipeline.generate(df)
    assert "F1" in res.columns
    assert (res["F1"] == 1).all()
    # Ensure original DF is not mutated
    assert "F1" not in df.columns

def test_pipeline_drop_na():
    pipeline = FeaturePipeline()
    pipeline.add_feature(DropAllFeature("DropMe"))
    df = pd.DataFrame({"close": [1, 2, 3]})
    res = pipeline.generate(df)
    assert len(res) == 1 # Only row 1 (the second row) doesn't have NA
