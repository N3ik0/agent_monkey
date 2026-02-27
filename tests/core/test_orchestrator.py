import pytest
import pandas as pd
from core.types import Action, MonkeySignal
from core.base_monkey import BaseMonkey
from core.orchestrator import MarketOrchestrator

class DummyMonkey(BaseMonkey):
    def __init__(self, name, action: Action, confidence: float, weight: float = 1.0, crash: bool = False):
        super().__init__(name, weight)
        self.action_to_return = action
        self.confidence_to_return = confidence
        self.crash = crash

    def analyze(self, market_data: pd.DataFrame) -> MonkeySignal:
        if self.crash:
            raise RuntimeError("Intentional crash")
        return MonkeySignal(self.name, self.action_to_return, self.confidence_to_return)

@pytest.fixture
def empty_df():
    return pd.DataFrame()

def test_orchestrator_initialization_empty():
    with pytest.raises(ValueError, match="requires at least one Monkey"):
        MarketOrchestrator([])

def test_orchestrator_consensus_buy(empty_df):
    m1 = DummyMonkey("M1", Action.BUY, 0.8)
    m2 = DummyMonkey("M2", Action.WAIT, 0.0)
    orch = MarketOrchestrator([m1, m2], activation_threshold=0.4)
    res = orch.get_consensus(empty_df)
    assert res["Signal"] == "BUY"
    assert res["Confiance"] == 0.4 # (0.8 * 1.0 + 0 * 1.0) / 2.0
    assert "Raw_Score" in res

def test_orchestrator_consensus_sell(empty_df):
    m1 = DummyMonkey("M1", Action.SELL, 0.9)
    orch = MarketOrchestrator([m1], activation_threshold=0.5)
    res = orch.get_consensus(empty_df)
    assert res["Signal"] == "SELL"
    assert res["Confiance"] == 0.9

def test_orchestrator_consensus_wait(empty_df): # Under threshold
    m1 = DummyMonkey("M1", Action.BUY, 0.3)
    orch = MarketOrchestrator([m1], activation_threshold=0.5)
    res = orch.get_consensus(empty_df)
    assert res["Signal"] == "WAIT"

def test_orchestrator_fail_fast(empty_df):
    m1 = DummyMonkey("CrashMonkey", Action.BUY, 0.5, crash=True)
    orch = MarketOrchestrator([m1])
    with pytest.raises(RuntimeError, match="Crash of agent 'CrashMonkey' during analysis: Intentional crash"):
        orch.get_consensus(empty_df)

def test_orchestrator_weighted_average(empty_df):
    m1 = DummyMonkey("M1", Action.BUY, 0.8, weight=2.0)
    m2 = DummyMonkey("M2", Action.SELL, 0.5, weight=1.0)
    orch = MarketOrchestrator([m1, m2], activation_threshold=0.1)
    # (1 * 0.8 * 2.0) + (-1 * 0.5 * 1.0) = 1.6 - 0.5 = 1.1 => 1.1 / 3.0 = 0.3666...
    res = orch.get_consensus(empty_df)
    assert res["Signal"] == "BUY"
    assert res["Confiance"] == 0.37
