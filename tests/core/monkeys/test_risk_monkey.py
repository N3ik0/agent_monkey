import pytest
import pandas as pd
from datetime import datetime, timezone
from core.types import Action, MonkeySignal, TradePlan
from core.monkeys.risk_monkey import RiskMonkey

@pytest.fixture
def base_risk_monkey():
    return RiskMonkey(name="TestRisk", atr_col="ATR_14", atr_multiplier=1.5, rr_ratio=2.0)

def test_analyze_no_atr(base_risk_monkey):
    df = pd.DataFrame({"close": [100.0], "ATR_14": [pd.NA]})
    signal = base_risk_monkey.analyze(df)
    assert signal.action == Action.WAIT
    assert signal.confidence == 0.0

def test_analyze_with_atr(base_risk_monkey):
    df = pd.DataFrame({"close": [100.0], "ATR_14": [2.0]})
    signal = base_risk_monkey.analyze(df)
    assert signal.action == Action.WAIT
    assert signal.confidence == 1.0

def test_analyze_missing_column(base_risk_monkey):
    df = pd.DataFrame({"close": [100.0]})
    with pytest.raises(KeyError, match="Missing column"):
        base_risk_monkey.analyze(df)

def test_compute_trade_plan_buy(base_risk_monkey):
    df = pd.DataFrame({"close": [100.0], "ATR_14": [2.0]})
    consensus = {
        "Signal": "BUY",
        "Confiance": 0.8,
        "Log_Agents": "Trend: BUY",
        "Raw_Score": 1.0
    }
    
    plan = base_risk_monkey.compute_trade_plan(df, consensus, "BTC/USDT")
    assert plan.direction == "BUY"
    assert plan.entry_price == 100.0
    # sl = 100 - (2.0 * 1.5) = 97.0
    assert plan.stop_loss == 97.0
    # tp = 100 + (2.0 * 1.5 * 2.0) = 106.0
    assert plan.take_profit == 106.0
    assert plan.confidence == 0.8
    assert "97.00" in plan.scenarios[0]
    assert "106.00" in plan.scenarios[0]

def test_compute_trade_plan_sell_low_confidence(base_risk_monkey):
    df = pd.DataFrame({"close": [100.0], "ATR_14": [2.0]})
    consensus = {
        "Signal": "SELL",
        "Confiance": 0.4,
        "Log_Agents": "Trend: SELL",
        "Raw_Score": -0.5
    }
    
    plan = base_risk_monkey.compute_trade_plan(df, consensus, "BTC/USDT")
    assert plan.direction == "SELL"
    # sl = 100 + (2.0 * 1.5) = 103.0
    assert plan.stop_loss == 103.0
    # tp = 100 - (2.0 * 1.5 * 2.0) = 94.0
    assert plan.take_profit == 94.0
    assert len(plan.scenarios) == 3
    assert "Confiance de l'orchestrateur faible" in plan.scenarios[2]

def test_compute_trade_plan_wait(base_risk_monkey):
    df = pd.DataFrame({"close": [100.0], "ATR_14": [pd.NA]})
    consensus = {"Signal": "WAIT", "Confiance": 0.0, "Log_Agents": ""}
    
    plan = base_risk_monkey.compute_trade_plan(df, consensus, "BTC/USDT")
    assert plan.direction == "WAIT"
    assert plan.stop_loss == 0.0
    assert plan.take_profit == 0.0
    assert "Aucun plan de trading" in plan.scenarios[0]
    
def test_trade_plan_to_markdown():
    plan = TradePlan(
        ticker="BTC/USDT",
        direction="BUY",
        entry_price=100.0,
        take_profit=110.0,
        stop_loss=95.0,
        risk_reward=2.0,
        confidence=0.9,
        atr_value=5.0,
        scenarios=["Scenario 1", "Scenario 2"],
        agents_log="[Trend: BUY]",
        generated_at="2024-01-01T00:00:00Z"
    )
    md = plan.to_markdown()
    assert "TradePlan" in md
    assert "BTC/USDT" in md
    assert "BUY" in md
    assert "100.0" in md
    assert "Scenario 1" in md
    assert "[Trend: BUY]" in md
