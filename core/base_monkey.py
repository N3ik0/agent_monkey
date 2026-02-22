from abc import ABC, abstractmethod
import pandas as pd

class BaseMonkey(ABC):
    """
    Interface parent (Abstract Base class) pour tous les agents de trading.
    Tout nouvel agent DOIT hériter de la classe mère
    """

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def analyze(self, market_data: pd.DataFrame) -> MonkeySignal:
        """
        Analyse les données de marchés et retourne une décision.
        :param market_data: DataFrame Pandas contenant les features nécessaires.
        :return: MonkeySignal (validé par core.types)
        """
        pass