import pandas as pd
from core.types import Action, MonkeySignal
from core.base_monkey import BaseMonkey


class MomentumMonkey(BaseMonkey):
    """
    A momentum-based trading agent that generates signals from the RSI indicator.
    Uses classic RSI overbought/oversold thresholds to identify potential
    reversals and continuation patterns.
    """

    def __init__(
        self,
        name: str,
        rsi_col: str = "RSI_14",
        overbought: float = 70.0,
        oversold: float = 30.0,
        weight: float = 1.0,
    ):
        """
        Initializes the MomentumMonkey.

        Args:
            name (str): Unique name for this agent instance.
            rsi_col (str): The RSI column name in the DataFrame (default 'RSI_14').
            overbought (float): RSI threshold above which the asset is overbought (SELL signal).
            oversold (float): RSI threshold below which the asset is oversold (BUY signal).
            weight (float): Weight of this agent in the orchestrator consensus.
        """
        super().__init__(name, weight)
        self.rsi_col = rsi_col
        self.overbought = overbought
        self.oversold = oversold

    def analyze(self, market_data: pd.DataFrame) -> MonkeySignal:
        """
        Analyzes market data using RSI to generate a trading signal.

        Logic:
            - RSI < oversold (30) → BUY (asset is undervalued)
            - RSI > overbought (70) → SELL (asset is overvalued)
            - Otherwise → WAIT

        Confidence is proportional to how far RSI is from the neutral zone.

        Args:
            market_data (pd.DataFrame): DataFrame containing the RSI column.

        Returns:
            MonkeySignal: The agent's decision with action and confidence.

        Raises:
            KeyError: If the required RSI column is missing.
        """
        if self.rsi_col not in market_data.columns:
            raise KeyError(
                f"Fatal error: Missing column '{self.rsi_col}' for {self.name}. "
                f"Ensure RSIFeature is in the FeaturePipeline."
            )

        latest = market_data.iloc[-1]

        # If RSI is NaN, we cannot make a decision
        if pd.isna(latest[self.rsi_col]):
            return MonkeySignal(self.name, Action.WAIT, 0.0)

        rsi_value: float = latest[self.rsi_col]

        if rsi_value < self.oversold:
            # The further below 30, the more confident the BUY signal
            # RSI 0 → confidence 1.0, RSI 30 → confidence 0.0
            confidence = min((self.oversold - rsi_value) / self.oversold, 1.0)
            return MonkeySignal(self.name, Action.BUY, confidence)

        elif rsi_value > self.overbought:
            # The further above 70, the more confident the SELL signal
            # RSI 100 → confidence 1.0, RSI 70 → confidence 0.0
            max_range = 100.0 - self.overbought
            confidence = min((rsi_value - self.overbought) / max_range, 1.0)
            return MonkeySignal(self.name, Action.SELL, confidence)

        else:
            return MonkeySignal(self.name, Action.WAIT, 0.0)
