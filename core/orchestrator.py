from typing import List, Dict, Any
import pandas as pd

from core.types import Action, MonkeySignal
from core.base_monkey import BaseMonkey

class MarketOrchestrator:
    """
    The Brain of the Agent-Monkey system.
    Aggregates signals from multiple agents to reach a market consensus.
    """

    def __init__(self, monkeys: List[BaseMonkey], activation_threshold: float = 0.4):
        """
        Initializes the Orchestrator with a set of trading agents.

        Args:
            monkeys (List[BaseMonkey]): List of agents (instances inheriting from BaseMonkey).
            activation_threshold (float): Minimum conviction threshold to trigger a BUY/SELL (0.0 to 1.0).
        """
        if not monkeys:
            raise ValueError("The Orchestrator requires at least one Monkey to operate.")
        
        self.monkeys = monkeys
        self.activation_threshold = activation_threshold

    def get_consensus(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Queries all agents and calculates the final signal.
        Formatted specifically for export to the Notion API (Market Sentinel).
        
        Args:
            market_data (pd.DataFrame): Pandas DataFrame containing historical data and features.
            
        Returns:
            Dict[str, Any]: Dictionary representing the final decision and agent logs.
        """
        signals: List[MonkeySignal] = []
        
        # 1. Signal Collection (Fail Fast if an agent crashes)
        for monkey in self.monkeys:
            try:
                # Pass a copy of the DataFrame to prevent Data Leakage between agents
                signal = monkey.analyze(market_data.copy())
                signals.append(signal)
            except Exception as e:
                # In production, we would log the error. Here we raise an exception to fix it early.
                raise RuntimeError(f"Crash of agent '{monkey.name}' during analysis: {str(e)}")

        # 2. Consensus Calculation (Weighted Average)
        total_weight = sum(monkey.weight for monkey in self.monkeys)
        
        weighted_score = 0.0
        logs_notion = []

        for monkey, signal in zip(self.monkeys, signals):
            # Formula: Action value (-1, 0, 1) * Agent confidence * Agent weight
            score = signal.action.value * signal.confidence * monkey.weight
            weighted_score += score
            
            # Prepare textual log for the 'Agent Logs' column in Notion
            logs_notion.append(f"[{monkey.name}: {signal.action.name} ({signal.confidence:.0%})]")

        # Normalize final score between -1.0 and 1.0
        final_score = weighted_score / total_weight
        abs_confidence = abs(final_score)

        # 3. Final Decision
        final_action = Action.WAIT
        if abs_confidence >= self.activation_threshold:
            final_action = Action.BUY if final_score > 0 else Action.SELL

        # 4. Strict formatting for the Notion Dashboard (Market Sentinel)
        return {
            "Signal": final_action.name,
            "Confiance": round(abs_confidence, 2),
            "Log_Agents": " | ".join(logs_notion),
            "Raw_Score": round(final_score, 4)
        }