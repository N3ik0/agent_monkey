import pandas as pd
from core.types import Action, MonkeySignal
from core.base_monkey import BaseMonkey


class MomentumMonkey(BaseMonkey):
    """
    A momentum-based trading agent that generates signals combining RSI and MACD.
    - BUY: RSI > 50 & MACD > 0 & MACD > Signal Line
    - SELL: RSI < 50 & MACD < 0 & MACD < Signal Line
    """

    def __init__(
        self,
        name: str,
        rsi_col: str = "RSI_14",
        macd_col: str = "MACD_line",
        macd_signal_col: str = "MACD_signal",
        atr_col: str = "ATR_14",
        weight: float = 1.0,
    ):
        """
        Initializes the MomentumMonkey.

        Args:
            name (str): Unique name for this agent instance.
            rsi_col (str): The RSI column name in the DataFrame (default 'RSI_14').
            macd_col (str): The MACD value column (default 'MACD_line').
            macd_signal_col (str): The MACD signal column (default 'MACD_signal').
            atr_col (str): The ATR column to normalize confidence (default 'ATR_14').
            weight (float): Weight of this agent in the orchestrator consensus.
        """
        super().__init__(name, weight)
        self.rsi_col = rsi_col
        self.macd_col = macd_col
        self.macd_signal_col = macd_signal_col
        self.atr_col = atr_col

    def analyze(self, market_data: pd.DataFrame) -> MonkeySignal:
        """
        Analyzes market data using RSI and MACD to generate a trading signal.
        """
        required_cols = [self.rsi_col, self.macd_col, self.macd_signal_col]
        missing = [c for c in required_cols if c not in market_data.columns]
        if missing:
            raise KeyError(f"Fatal error: Missing columns {missing} for {self.name}.")

        latest = market_data.iloc[-1]

        # If core indicators are NaN, wait
        if pd.isna(latest[self.rsi_col]) or pd.isna(latest[self.macd_col]) or pd.isna(latest[self.macd_signal_col]):
            return MonkeySignal(self.name, Action.WAIT, 0.0)

        rsi: float = float(latest[self.rsi_col])
        macd: float = float(latest[self.macd_col])
        macd_signal: float = float(latest[self.macd_signal_col])
        
        atr_val = latest.get(self.atr_col, pd.NA)
        atr = float(atr_val) if not pd.isna(atr_val) else pd.NA

        # Logic checks
        is_buy = (rsi > 50) and (macd > 0) and (macd > macd_signal)
        is_sell = (rsi < 50) and (macd < 0) and (macd < macd_signal)

        if not is_buy and not is_sell:
            return MonkeySignal(self.name, Action.WAIT, 0.0)

        # Confidence calculation
        rsi_strength = min(abs(rsi - 50) / 50.0, 1.0)
        
        if pd.isna(atr) or atr <= 0:
            macd_strength = 0.5
        else:
            macd_strength = min(abs(macd) / (atr * 0.1), 1.0)
            
        confidence = (rsi_strength + macd_strength) / 2.0

        if is_buy:
            return MonkeySignal(self.name, Action.BUY, confidence)
        else:
            return MonkeySignal(self.name, Action.SELL, confidence)
