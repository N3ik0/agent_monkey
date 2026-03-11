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
        "Slow": [100, 100],
        "close": [100, 110] # Price above Slow (validates BUY)
    })
    signal = monkey.analyze(df)
    assert signal.action == Action.BUY
    # diff = 5 / 100 = 0.05. confidence = min(0.05 * 20, 1.0) = 1.0
    assert signal.confidence == 1.0

def test_trend_monkey_sell():
    monkey = TrendMonkey("TestTrend", fast_col="Fast", slow_col="Slow")
    df = pd.DataFrame({
        "Fast": [100, 95],
        "Slow": [100, 100],
        "close": [100, 90] # Price below Slow (validates SELL)
    })
    signal = monkey.analyze(df)
    assert signal.action == Action.SELL
    # diff = 5 / 100 = 0.05. confidence = min(0.05 * 20, 1.0) = 1.0
    assert signal.confidence == 1.0

def test_trend_monkey_momentum_filter():
    monkey = TrendMonkey("TestTrend", fast_col="Fast", slow_col="Slow")
    # Spread is 100.2 - 100 = 0.2 -> diff = 0.002. Less than 0.003
    df = pd.DataFrame({
        "Fast": [100, 100.2],
        "Slow": [100, 100.0],
        "close": [100, 110] # Good price, but momentum too weak
    })
    signal = monkey.analyze(df)
    assert signal.action == Action.WAIT
    assert signal.confidence == 0.0
    
def test_trend_monkey_price_filter_buy():
    monkey = TrendMonkey("TestTrend", fast_col="Fast", slow_col="Slow")
    df = pd.DataFrame({
        "Fast": [100, 105], # Bull crossover
        "Slow": [100, 100],
        "close": [100, 98]  # Bearish unconfirmed price
    })
    signal = monkey.analyze(df)
    assert signal.action == Action.WAIT
    
def test_trend_monkey_price_filter_sell():
    monkey = TrendMonkey("TestTrend", fast_col="Fast", slow_col="Slow")
    df = pd.DataFrame({
        "Fast": [100, 95], # Bear crossover
        "Slow": [100, 100],
        "close": [100, 102]  # Bullish unconfirmed price
    })
    signal = monkey.analyze(df)
    assert signal.action == Action.WAIT

def test_trend_monkey_nan_values():
    monkey = TrendMonkey("TestTrend", fast_col="Fast", slow_col="Slow")
    df = pd.DataFrame({
        "Fast": [pd.NA],
        "Slow": [pd.NA],
        "close": [pd.NA]
    })
    signal = monkey.analyze(df)
    assert signal.action == Action.WAIT

def test_trend_monkey_missing_columns():
    monkey = TrendMonkey("TestTrend", fast_col="Fast", slow_col="Slow")
    df = pd.DataFrame({"close": [1, 2]})
    with pytest.raises(KeyError, match="Missing columns"):
        monkey.analyze(df)
