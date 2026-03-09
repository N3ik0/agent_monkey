"""Tests for MomentumMonkey."""
import pytest
import pandas as pd
from core.types import Action
from core.monkeys.momentum_monkey import MomentumMonkey


@pytest.fixture
def monkey() -> MomentumMonkey:
    """Creates a default MomentumMonkey instance."""
    return MomentumMonkey(name="TestMomentum", rsi_col="RSI_14")


def test_momentum_monkey_buy(monkey: MomentumMonkey) -> None:
    """RSI below 30 should produce a BUY signal."""
    df = pd.DataFrame({"RSI_14": [50.0, 45.0, 20.0]})
    signal = monkey.analyze(df)

    assert signal.action == Action.BUY
    assert signal.confidence > 0.0
    assert signal.confidence <= 1.0


def test_momentum_monkey_sell(monkey: MomentumMonkey) -> None:
    """RSI above 70 should produce a SELL signal."""
    df = pd.DataFrame({"RSI_14": [50.0, 65.0, 85.0]})
    signal = monkey.analyze(df)

    assert signal.action == Action.SELL
    assert signal.confidence > 0.0
    assert signal.confidence <= 1.0


def test_momentum_monkey_wait(monkey: MomentumMonkey) -> None:
    """RSI in the neutral zone (30-70) should produce a WAIT signal."""
    df = pd.DataFrame({"RSI_14": [50.0, 55.0, 45.0]})
    signal = monkey.analyze(df)

    assert signal.action == Action.WAIT
    assert signal.confidence == 0.0


def test_momentum_monkey_extreme_buy(monkey: MomentumMonkey) -> None:
    """RSI at 0 should produce maximum BUY confidence."""
    df = pd.DataFrame({"RSI_14": [0.0]})
    signal = monkey.analyze(df)

    assert signal.action == Action.BUY
    assert signal.confidence == 1.0


def test_momentum_monkey_extreme_sell(monkey: MomentumMonkey) -> None:
    """RSI at 100 should produce maximum SELL confidence."""
    df = pd.DataFrame({"RSI_14": [100.0]})
    signal = monkey.analyze(df)

    assert signal.action == Action.SELL
    assert signal.confidence == 1.0


def test_momentum_monkey_nan_values(monkey: MomentumMonkey) -> None:
    """NaN RSI should produce a WAIT signal with 0 confidence."""
    df = pd.DataFrame({"RSI_14": [float("nan")]})
    signal = monkey.analyze(df)

    assert signal.action == Action.WAIT
    assert signal.confidence == 0.0


def test_momentum_monkey_missing_column(monkey: MomentumMonkey) -> None:
    """Missing RSI column should raise KeyError."""
    df = pd.DataFrame({"close": [100.0, 101.0]})
    with pytest.raises(KeyError, match="RSI_14"):
        monkey.analyze(df)
