import pytest
import pandas as pd
from core.types import Action
from core.monkeys.trend_monkey import TrendMonkey

@pytest.fixture
def empty_monkey():
    return TrendMonkey("TestTrend", fast_col="Fast", slow_col="Slow")

def test_trend_monkey_buy():
    monkey = TrendMonkey("TestTrend", fast_col="Fast", slow_col="Slow")
    df = pd.DataFrame({
        "Fast": [100, 105],
        "Slow": [100, 100]
    })
    signal = monkey.analyze(df)
    assert signal.action == Action.BUY
    # diff = 5 / 100 = 0.05. confidence = min(0.05 * 20, 1.0) = 1.0
    assert signal.confidence == 1.0

def test_trend_monkey_sell():
    monkey = TrendMonkey("TestTrend", fast_col="Fast", slow_col="Slow")
    df = pd.DataFrame({
        "Fast": [100, 95],
        "Slow": [100, 100]
    })
    signal = monkey.analyze(df)
    assert signal.action == Action.SELL
    # diff = 5 / 100 = 0.05. confidence = min(0.05 * 20, 1.0) = 1.0
    assert signal.confidence == 1.0

def test_trend_monkey_wait():
    monkey = TrendMonkey("TestTrend", fast_col="Fast", slow_col="Slow")
    df = pd.DataFrame({
        "Fast": [100, 100],
        "Slow": [100, 100]
    })
    signal = monkey.analyze(df)
    assert signal.action == Action.WAIT
    assert signal.confidence == 0.0

def test_trend_monkey_nan_values():
    monkey = TrendMonkey("TestTrend", fast_col="Fast", slow_col="Slow")
    df = pd.DataFrame({
        "Fast": [pd.NA],
        "Slow": [pd.NA]
    })
    signal = monkey.analyze(df)
    assert signal.action == Action.WAIT

def test_trend_monkey_missing_columns():
    monkey = TrendMonkey("TestTrend", fast_col="Fast", slow_col="Slow")
    df = pd.DataFrame({"close": [1, 2]})
    with pytest.raises(KeyError, match="Missing columns"):
        monkey.analyze(df)
