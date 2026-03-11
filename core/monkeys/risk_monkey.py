import pandas as pd
from datetime import datetime, timezone

from core.base_monkey import BaseMonkey
from core.types import Action, MonkeySignal, TradePlan

class RiskMonkey(BaseMonkey):
    """
    Risk Management agent.
    It doesn't provide directional signals (BUY/SELL) but evaluates the risk metrics
    (Stop-Loss, Take-Profit, Risk/Reward) using the Average True Range (ATR).
    """

    def __init__(
        self,
        name: str = "RiskMonkey",
        atr_col: str = "ATR_14",
        atr_multiplier: float = 1.5,
        rr_ratio: float = 2.0,
        weight: float = 1.0,
    ):
        super().__init__(name, weight)
        self.atr_col = atr_col
        self.atr_multiplier = atr_multiplier
        self.rr_ratio = rr_ratio

    def analyze(self, market_data: pd.DataFrame) -> MonkeySignal:
        """
        RiskMonkey always waits, but its confidence depends on the availability
        of the ATR to compute the risk.
        """
        if self.atr_col not in market_data.columns:
            raise KeyError(
                f"Fatal error: Missing column for {self.name}. Required: {self.atr_col}"
            )

        latest = market_data.iloc[-1]
        
        if pd.isna(latest[self.atr_col]):
            return MonkeySignal(self.name, Action.WAIT, 0.0)
            
        return MonkeySignal(self.name, Action.WAIT, 1.0)

    def compute_trade_plan(self, df: pd.DataFrame, consensus: dict, ticker: str) -> TradePlan:
        """
        Generates a structured TradePlan based on the orchestrator's consensus
        and the current market volatility (ATR).
        
        Args:
            df (pd.DataFrame): The market data DataFrame.
            consensus (dict): The dictionary returned by MarketOrchestrator.get_consensus()
            ticker (str): The asset symbol.
            
        Returns:
            TradePlan: The calculated trade parameters.
        """
        latest = df.iloc[-1]
        entry = float(latest["close"])
        raw_atr = latest.get(self.atr_col, pd.NA)
        atr = float(raw_atr) if not pd.isna(raw_atr) else pd.NA
        
        direction = consensus.get("Signal", "WAIT")
        conf = consensus.get("Confiance", 0.0)
        logs = consensus.get("Log_Agents", "")
        
        scenarios = []
        tp = 0.0
        sl = 0.0
        rr = 0.0
        
        if direction == "BUY" and not pd.isna(atr):
            sl = entry - (atr * self.atr_multiplier)
            tp = entry + (atr * self.atr_multiplier * self.rr_ratio)
            rr = (tp - entry) / (entry - sl) if sl != entry else 0.0
            
            scenarios.append(f"Si le prix repasse sous {sl:.2f} avant d'atteindre {tp:.2f} → signal invalidé.")
            scenarios.append(f"Prise de profit partielle recommandée à la moitié du mouvement ({(entry + tp) / 2:.2f}).")
            if conf < 0.6:
                scenarios.append("Confiance de l'orchestrateur faible : Mettre un Stop Loss suiveur rapidement.")
                
        elif direction == "SELL" and not pd.isna(atr):
            sl = entry + (atr * self.atr_multiplier)
            tp = entry - (atr * self.atr_multiplier * self.rr_ratio)
            rr = (entry - tp) / (sl - entry) if sl != entry else 0.0
            
            scenarios.append(f"Si le prix remonte au-dessus de {sl:.2f} avant d'atteindre {tp:.2f} → signal invalidé.")
            scenarios.append(f"Prise de profit partielle recommandée à la moitié du mouvement ({(entry + tp) / 2:.2f}).")
            if conf < 0.6:
                scenarios.append("Confiance de l'orchestrateur faible : Réduire l'exposition ou resserrer le SL.")
                
        else:
            # WAIT ou ATR manquant
            scenarios.append("Aucun plan de trading actif. En attente de signaux clairs ou de données de volatilité (ATR).")

        return TradePlan(
            ticker=ticker,
            direction=direction,
            entry_price=entry,
            take_profit=tp,
            stop_loss=sl,
            risk_reward=rr,
            confidence=conf,
            atr_value=atr if not pd.isna(atr) else 0.0,
            scenarios=scenarios,
            agents_log=logs,
            generated_at=datetime.now(timezone.utc).isoformat()
        )
