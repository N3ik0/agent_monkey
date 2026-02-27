import pytest
from core.types import Action, MonkeySignal

def test_monkey_signal_valid():
    signal = MonkeySignal(monkey_name="Test", action=Action.BUY, confidence=0.5)
    assert signal.confidence == 0.5
    assert signal.action == Action.BUY

def test_monkey_signal_invalid_high_confidence():
    with pytest.raises(ValueError, match="Confidence should be between 0.0 and 1.0"):
        MonkeySignal(monkey_name="Test", action=Action.BUY, confidence=1.5)

def test_monkey_signal_invalid_low_confidence():
    with pytest.raises(ValueError, match="Confidence should be between 0.0 and 1.0"):
        MonkeySignal(monkey_name="Test", action=Action.BUY, confidence=-0.1)
