from abc import ABC, abstractmethod
import pandas as pd
from core.types import MonkeySignal

class BaseMonkey(ABC):
    """
    Parent interface (Abstract Base Class) for all trading agents.
    Any new agent MUST inherit from the base class.
    """

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def analyze(self, market_data: pd.DataFrame) -> MonkeySignal:
        """
        Analyzes market data and returns a decision.
        :param market_data: Pandas DataFrame containing the necessary features.
        :return: MonkeySignal (validated by core.types)
        """
        pass