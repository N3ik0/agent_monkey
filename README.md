# 🐒 Agent-Monkey — Multi-Agent System for Quantitative Trading

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-44%20passed-brightgreen.svg)](#tests)

Un système multi-agents (MAS) modulaire où des agents spécialisés (_Monkeys_) analysent des données de marché et génèrent un consensus de trading via un orchestrateur central.

---

## 📐 Architecture

```
agent_monkey/
├── core/                     # Fondation immuable
│   ├── types.py              # Action (BUY/SELL/WAIT), MonkeySignal
│   ├── base_monkey.py        # ABC — interface pour tous les agents
│   ├── orchestrator.py       # MarketOrchestrator — agrège les signaux
│   └── monkeys/              # Agents de trading
│       ├── trend_monkey.py   # Agent suivi de tendance (SMA crossover)
│       └── momentum_monkey.py# Agent momentum (RSI overbought/oversold)
│
├── features/                 # Feature Factory (indicateurs techniques)
│   ├── base_feature.py       # ABC — interface pour tous les indicateurs
│   ├── pipeline.py           # FeaturePipeline — chaîne de calcul
│   └── technical/
│       ├── sma.py            # Simple Moving Average
│       ├── ema.py            # Exponential Moving Average
│       ├── rsi.py            # Relative Strength Index
│       ├── macd.py           # MACD (line, signal, histogram)
│       └── bollinger.py      # Bollinger Bands (upper, middle, lower)
│
├── data/                     # Acquisition de données
│   ├── base_fetcher.py       # ABC — interface pour les fetchers
│   └── yfinance_fetcher.py   # Implémentation Yahoo Finance
│
├── tests/                    # Tests (pytest) — miroir de la structure source
├── main.py                   # Script de backtest end-to-end
└── README.md
```

### Flux de données

```
Yahoo Finance → DataFetcher → FeaturePipeline → [TrendMonkey, MomentumMonkey] → Orchestrator → Consensus
```

---

## 🚀 Installation

```bash
# Cloner le projet
git clone <repo-url> agent_monkey
cd agent_monkey

# Créer et activer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate     # Linux/Mac
# .venv\Scripts\activate      # Windows

# Installer les dépendances
pip install pandas numpy yfinance pytest
```

---

## ▶️ Utilisation

### Lancer un backtest

```bash
# Backtest BTC-USD sur les 30 derniers jours (défaut)
python main.py

# Backtest sur un ticker spécifique
python main.py --ticker AAPL --period 1y --lookback 60

# Toutes les options
python main.py --help
```

| Argument     | Défaut    | Description                            |
|--------------|-----------|----------------------------------------|
| `--ticker`   | `BTC-USD` | Symbole de l'actif                     |
| `--period`   | `6mo`     | Période de données (`1mo`, `6mo`, `1y`)|
| `--interval` | `1d`      | Intervalle des bougies (`1d`, `1h`)    |
| `--lookback` | `30`      | Nombre de jours à simuler             |

### Exemple de sortie

```
📡 Fetched 124 candles for BTC-USD (6mo, 1d)
🔧 Features computed: ['SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI_14', 'MACD_12_26_9', 'BB_20']
📊 Rows after NaN cleanup: 74

======================================================================
  BACKTEST: BTC-USD — Last 30 trading days
======================================================================

Date           Signal   Confidence   Score      Agents Log
-------------- -------- ------------ ---------- ----------------------------------------
2026-02-07     🟢 BUY  0.65         0.6500     [TrendMonkey: BUY (80%)] | [MomentumMonkey: WAIT (0%)]
2026-02-08     ⚪ WAIT 0.12         0.1200     [TrendMonkey: BUY (25%)] | [MomentumMonkey: WAIT (0%)]
...

======================================================================
  SUMMARY: 🟢 BUY=12  🔴 SELL=5  ⚪ WAIT=13
  Latest Signal: WAIT (Confidence: 0.15)
======================================================================
```

---

## 🧪 Tests

```bash
# Lancer tous les tests
python -m pytest tests/ -v

# Lancer les tests d'un module spécifique
python -m pytest tests/features/technical/test_macd.py -v
python -m pytest tests/core/monkeys/test_momentum_monkey.py -v
```

Les tests incluent :
- **Validation des signaux** — BUY/SELL/WAIT selon les seuils
- **Anti-look-ahead bias** — vérifie que les features n'utilisent pas de données futures
- **Colonnes manquantes** — crash explicite (Fail Fast)
- **Valeurs NaN** — gestion correcte des données manquantes
- **Mocks API** — aucun appel réseau dans les tests

---

## 🧩 Étendre le système

### Ajouter un nouvel indicateur

Créer un fichier dans `features/technical/` :

```python
from features.base_feature import BaseFeature
import pandas as pd

class MyFeature(BaseFeature):
    def __init__(self, window: int = 14):
        super().__init__(name=f"MY_INDICATOR_{window}")
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if "close" not in df.columns:
            raise KeyError("Fatal error: Column 'close' missing.")
        df[self.name] = df["close"].rolling(self.window).mean()  # exemple
        return df
```

Puis l'ajouter dans `main.py` → `build_pipeline()`.

### Ajouter un nouvel agent

Créer un fichier dans `core/monkeys/` :

```python
from core.base_monkey import BaseMonkey
from core.types import Action, MonkeySignal
import pandas as pd

class MyMonkey(BaseMonkey):
    def analyze(self, market_data: pd.DataFrame) -> MonkeySignal:
        # Votre logique ici
        return MonkeySignal(self.name, Action.BUY, 0.75)
```

Puis l'ajouter dans `main.py` → `build_orchestrator()`.

---

## 📋 Principes de conception

| Principe | Application |
|---|---|
| **SRP** | Un agent analyse, il ne fetch pas de données |
| **OCP** | Ajouter un Monkey/Feature sans toucher à l'Orchestrator |
| **DIP** | Tout dépend des abstractions (`BaseMonkey`, `BaseFeature`) |
| **Fail Fast** | Crash explicite sur données corrompues |
| **Immutabilité** | Les DataFrames sont copiés avant distribution aux agents |
| **No Look-Ahead** | Interdit d'utiliser `shift(-x)` dans les features |
