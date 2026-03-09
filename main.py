"""
Agent-Monkey — Main Backtest Script
====================================
End-to-end pipeline: Fetch data → Compute features → Run agents → Display consensus.

Usage:
    python main.py --ticker BTC-USD --period 6mo
    python main.py --ticker AAPL --period 1y --interval 1d
"""

import argparse
from typing import List

import pandas as pd

from data.yfinance_fetcher import YFinanceDataFetcher
from features.pipeline import FeaturePipeline
from features.technical.sma import SMAFeature
from features.technical.ema import EMAFeature
from features.technical.rsi import RSIFeature
from features.technical.macd import MACDFeature
from features.technical.bollinger import BollingerFeature
from core.orchestrator import MarketOrchestrator
from core.monkeys.trend_monkey import TrendMonkey
from core.monkeys.momentum_monkey import MomentumMonkey


def build_pipeline() -> FeaturePipeline:
    """
    Constructs the feature pipeline with all active indicators.

    Returns:
        FeaturePipeline: A configured pipeline ready to process raw OHLCV data.
    """
    pipeline = FeaturePipeline()
    pipeline.add_feature(SMAFeature(window=20))
    pipeline.add_feature(SMAFeature(window=50))
    pipeline.add_feature(EMAFeature(window=12))
    pipeline.add_feature(EMAFeature(window=26))
    pipeline.add_feature(RSIFeature(window=14))
    pipeline.add_feature(MACDFeature())
    pipeline.add_feature(BollingerFeature(window=20))
    return pipeline


def build_orchestrator() -> MarketOrchestrator:
    """
    Constructs the orchestrator with the available trading agents.

    Returns:
        MarketOrchestrator: A configured orchestrator with all active Monkeys.
    """
    monkeys = [
        TrendMonkey(name="TrendMonkey", fast_col="SMA_20", slow_col="SMA_50", weight=1.0),
        MomentumMonkey(name="MomentumMonkey", rsi_col="RSI_14", weight=1.0),
    ]
    return MarketOrchestrator(monkeys=monkeys, activation_threshold=0.4)


def run_backtest(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
    lookback: int = 30,
) -> List[dict]:
    """
    Runs a simple backtest: iterates over the last N trading days
    and collects the orchestrator's consensus for each day.

    Args:
        ticker (str): The asset symbol (e.g., 'BTC-USD').
        period (str): The data period to fetch (e.g., '6mo', '1y').
        interval (str): The candle interval (e.g., '1d').
        lookback (int): Number of recent days to simulate signals for.

    Returns:
        List[dict]: A list of consensus dictionaries, one per simulated day.
    """
    # 1. Fetch raw data
    fetcher = YFinanceDataFetcher()
    raw_df = fetcher.fetch(ticker=ticker, period=period, interval=interval)
    print(f"📡 Fetched {len(raw_df)} candles for {ticker} ({period}, {interval})")

    # 2. Compute features
    pipeline = build_pipeline()
    processed_df = pipeline.generate(raw_df)
    print(f"🔧 Features computed: {pipeline.get_feature_name()}")
    print(f"📊 Rows after NaN cleanup: {len(processed_df)}")

    if len(processed_df) < 2:
        raise ValueError(
            f"Not enough data after feature computation. "
            f"Only {len(processed_df)} rows remain. Try a longer period."
        )

    # 3. Build orchestrator
    orchestrator = build_orchestrator()

    # 4. Simulate signals for the last N days
    effective_lookback = min(lookback, len(processed_df) - 1)
    results: List[dict] = []

    print(f"\n{'='*70}")
    print(f"  BACKTEST: {ticker} — Last {effective_lookback} trading days")
    print(f"{'='*70}\n")
    print(f"{'Date':<14} {'Signal':<8} {'Confidence':<12} {'Score':<10} {'Agents Log'}")
    print(f"{'-'*14} {'-'*8} {'-'*12} {'-'*10} {'-'*40}")

    for i in range(effective_lookback, 0, -1):
        # Slice up to row index to avoid look-ahead bias
        slice_end = len(processed_df) - i + 1
        data_slice = processed_df.iloc[:slice_end]
        date_label = str(data_slice.index[-1].date()) if hasattr(data_slice.index[-1], 'date') else str(data_slice.index[-1])

        consensus = orchestrator.get_consensus(data_slice)
        consensus["Date"] = date_label
        results.append(consensus)

        # Color-coded signal
        signal = consensus["Signal"]
        if signal == "BUY":
            signal_display = f"🟢 {signal}"
        elif signal == "SELL":
            signal_display = f"🔴 {signal}"
        else:
            signal_display = f"⚪ {signal}"

        print(
            f"{date_label:<14} {signal_display:<8} "
            f"{consensus['Confiance']:<12.2f} "
            f"{consensus['Raw_Score']:<10.4f} "
            f"{consensus['Log_Agents']}"
        )

    # 5. Summary
    buy_count = sum(1 for r in results if r["Signal"] == "BUY")
    sell_count = sum(1 for r in results if r["Signal"] == "SELL")
    wait_count = sum(1 for r in results if r["Signal"] == "WAIT")

    print(f"\n{'='*70}")
    print(f"  SUMMARY: 🟢 BUY={buy_count}  🔴 SELL={sell_count}  ⚪ WAIT={wait_count}")
    print(f"  Latest Signal: {results[-1]['Signal']} (Confidence: {results[-1]['Confiance']:.2f})")
    print(f"{'='*70}")

    return results


def main() -> None:
    """
    CLI entry point for the Agent-Monkey backtest system.
    """
    parser = argparse.ArgumentParser(
        description="Agent-Monkey MAS — Backtest Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ticker", type=str, default="BTC-USD",
        help="Asset ticker symbol (default: BTC-USD)"
    )
    parser.add_argument(
        "--period", type=str, default="6mo",
        help="Data period to fetch (default: 6mo)"
    )
    parser.add_argument(
        "--interval", type=str, default="1d",
        help="Candle interval (default: 1d)"
    )
    parser.add_argument(
        "--lookback", type=int, default=30,
        help="Number of recent days to simulate (default: 30)"
    )

    args = parser.parse_args()

    run_backtest(
        ticker=args.ticker,
        period=args.period,
        interval=args.interval,
        lookback=args.lookback,
    )


if __name__ == "__main__":
    main()
