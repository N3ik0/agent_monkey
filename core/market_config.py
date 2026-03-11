import yaml
from pathlib import Path

class MarketConfig:
    """
    Handles loading of market-specific configuration from YAML files.
    Allows for dynamic tuning of agents and rules (like ATR multiplier, SMA windows)
    based on the asset being traded.
    """

    @staticmethod
    def load(ticker: str) -> dict:
        """
        Loads the YAML configuration for the given ticker.
        If a specific file is not found, falls back to default.yaml.
        
        Args:
            ticker (str): The asset symbol (e.g. BTC/USDT, EURUSD, AAPL).

        Returns:
            dict: The loaded configuration dictionary.
        """
        sanitized_ticker = ticker.replace("/", "_").replace("-", "_")
        config_dir = Path("config")
        
        target_path = config_dir / f"{sanitized_ticker}.yaml"
        default_path = config_dir / "default.yaml"

        loaded_config = {}

        # 1. Always load default config first as the fallback base
        if default_path.exists():
            with open(default_path, "r", encoding="utf-8") as f:
                loaded_config.update(yaml.safe_load(f) or {})
        else:
            raise FileNotFoundError("Fatal Error: config/default.yaml is missing.")

        # 2. If a market-specific config exists, overwrite the defaults
        if target_path.exists():
            print(f"🔧 Loaded specific market config: {target_path.name}")
            with open(target_path, "r", encoding="utf-8") as f:
                specific_config = yaml.safe_load(f) or {}
                loaded_config.update(specific_config)
        else:
            print(f"🔧 No exact config for {ticker}, using default.yaml")

        return loaded_config
