import pytest
import pandas as pd
from core.types import Action
from core.monkeys.momentum_monkey import MomentumMonkey

@pytest.fixture
def monkey() -> MomentumMonkey:
    """Creates a default MomentumMonkey instance."""
    return MomentumMonkey(name="TestMomentum")

def test_momentum_monkey_buy(monkey: MomentumMonkey) -> None:
    """RSI > 50, MACD > 0 and MACD > Signal Line should produce a BUY signal."""
    df = pd.DataFrame({
        "RSI_14": [60.0],
        "MACD_line": [1.5],
        "MACD_signal": [1.0],
        "ATR_14": [10.0]
    })
    signal = monkey.analyze(df)
    
    assert signal.action == Action.BUY
    # RSI strength: abs(60-50)/50 = 0.2
    # MACD strength: abs(1.5)/(10.0*0.1) = min(1.5/1.0, 1.0) = 1.0
    # Confidence: (0.2 + 1.0) / 2 = 0.6
    assert signal.confidence == 0.6

def test_momentum_monkey_sell(monkey: MomentumMonkey) -> None:
    """RSI < 50, MACD < 0 and MACD < Signal Line should produce a SELL signal."""
    df = pd.DataFrame({
        "RSI_14": [40.0],
        "MACD_line": [-1.5],
        "MACD_signal": [-1.0],
        "ATR_14": [10.0]
    })
    signal = monkey.analyze(df)
    
    assert signal.action == Action.SELL
    # RSI strength: abs(40-50)/50 = 0.2
    # MACD strength: abs(-1.5)/(10.0*0.1) = min(1.5/1.0, 1.0) = 1.0
    # Confidence: (0.2 + 1.0) / 2 = 0.6
    assert signal.confidence == 0.6

def test_momentum_monkey_wait_contradictory(monkey: MomentumMonkey) -> None:
    """RSI > 50 but MACD < 0 should produce a WAIT signal."""
    df = pd.DataFrame({
        "RSI_14": [60.0],
        "MACD_line": [-1.5],
        "MACD_signal": [-2.0],
        "ATR_14": [10.0]
    })
    signal = monkey.analyze(df)
    
    assert signal.action == Action.WAIT
    assert signal.confidence == 0.0

def test_momentum_monkey_crossover_fail(monkey: MomentumMonkey) -> None:
    """RSI > 50, MACD > 0 but MACD < Signal Line should produce a WAIT signal."""
    df = pd.DataFrame({
        "RSI_14": [60.0],
        "MACD_line": [1.0],
        "MACD_signal": [1.5],
        "ATR_14": [10.0]
    })
    signal = monkey.analyze(df)
    
    assert signal.action == Action.WAIT
    assert signal.confidence == 0.0

def test_momentum_monkey_missing_atr(monkey: MomentumMonkey) -> None:
    """Missing ATR should fallback to 0.5 for MACD strength."""
    df = pd.DataFrame({
        "RSI_14": [60.0],
        "MACD_line": [1.5],
        "MACD_signal": [1.0],
        "ATR_14": [pd.NA]
    })
    signal = monkey.analyze(df)
    
    assert signal.action == Action.BUY
    # RSI: 0.2, MACD: 0.5 -> Conf: 0.35
    assert signal.confidence == 0.35

def test_momentum_monkey_nan_values(monkey: MomentumMonkey) -> None:
    """NaN RSI/MACD should produce a WAIT signal with 0 confidence."""
    df = pd.DataFrame({
        "RSI_14": [pd.NA],
        "MACD_line": [1.0],
        "MACD_signal": [0.5],
        "ATR_14": [10.0]
    })
    signal = monkey.analyze(df)
    
    assert signal.action == Action.WAIT
    assert signal.confidence == 0.0

def test_momentum_monkey_missing_column(monkey: MomentumMonkey) -> None:
    """Missing MACD line column should raise KeyError."""
    df = pd.DataFrame({"RSI_14": [50.0], "MACD_signal": [1.0], "close": [100.0]})
    with pytest.raises(KeyError, match="Missing columns"):
        monkey.analyze(df)
