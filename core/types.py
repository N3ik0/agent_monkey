from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any

class Action(Enum):
    """
    Defines authorized trades.
    """
    BUY = 1
    SELL = -1
    WAIT = 0

@dataclass
class MonkeySignal:
    """
    Standard format for decision takes by agent (Monkey).
    Object send to the notion database.
    """
    monkey_name: str
    action: Action
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Quant validation: Automatic security executed at object creation.
        """
        if not(0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Erreur fatale ({self.monkey_name}) : "
                f"Confidence should be between 0.0 and 1.0, Receive: {self.confidence}"
            )