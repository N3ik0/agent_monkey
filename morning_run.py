import argparse
from typing import List

from data.fetcher_router import DataFetcherRouter
from core.types import TradePlan
from core.monkeys.risk_monkey import RiskMonkey
from core.market_config import MarketConfig
# Import the builders from main.py to avoid redefining the whole application stack
from main import build_pipeline, build_orchestrator

# Default configurable watchlist
WATCHLIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

def scan_market(watchlist: List[str], period: str = "6mo", interval: str = "1d") -> List[TradePlan]:
    """
    Scans the provided watchlist and generates TradePlans using the default MAS setup.
    Errors on single assets are caught and logged without breaking the whole process.
    """
    # 1. Fetch raw data via the Router (Fail-Fast inside router, caught here)
    router = DataFetcherRouter()
    
    # Instantiate RiskMonkey explicitly for TradePlan generation
    risk_monkey = RiskMonkey()
        
    plans = []
    
    print(f"\n{'='*60}")
    print(f"🌅 MORNING RUN - Scanning {len(watchlist)} assets")
    print(f"{'='*60}\n")
    
    for ticker in watchlist:
        print(f"📡 Processing {ticker}...", end=" ")
        try:
            config = MarketConfig.load(ticker)
            pipeline = build_pipeline(config=config)
            orchestrator = build_orchestrator(config=config)
            
            # 1. Fetch raw data via the Router (Fail-Fast inside router, caught here)
            raw_df = router.fetch(ticker=ticker, period=period, interval=interval)
            
            # 2. Compute features via Pipeline
            processed_df = pipeline.generate(raw_df)
            if len(processed_df) < 2:
                print(f"\n⚠️ Not enough data after feature computation. Skipping.")
                continue
                
            # 3. Get consensus bounds for the latest available day
            consensus = orchestrator.get_consensus(processed_df)
            
            # 4. Generate TradePlan metrics
            plan = risk_monkey.compute_trade_plan(processed_df, consensus, ticker)
            
            # Override generated_at with actual slice date for accuracy instead of execution time
            if hasattr(processed_df.index[-1], 'date'):
                plan.generated_at = str(processed_df.index[-1].date())
            else:
                plan.generated_at = str(processed_df.index[-1])
                
            plans.append(plan)
            
            # Minimal feedback in terminal
            signal = plan.direction
            color = "🟢 BUY" if signal == "BUY" else "🔴 SELL" if signal == "SELL" else "⚪ WAIT"
            print(f"✅ {color} (Conf: {plan.confidence:.2f})")
            
        except Exception as e:
            # Handle fetch or compute error without killing the scan
            print(f"\n❌ Failed: {str(e)}")
            
    return plans

def display_summary(plans: List[TradePlan]):
    """
    Displays a global tabular summary of all scanned assets,
    followed by the detailed TradePlans for actionable items.
    """
    print(f"\n{'='*85}")
    print(f"📈 GLOBAL SUMMARY")
    print(f"{'='*85}")
    
    t_w, s_w, c_w, p_w, tp_w, sl_w, rr_w = 12, 8, 10, 10, 10, 10, 8
    
    print(f"{'Ticker':<{t_w}} | {'Signal':<{s_w}} | {'Conf':<{c_w}} | {'Entry':<{p_w}} | {'TP':<{tp_w}} | {'SL':<{sl_w}} | {'R:R':<{rr_w}}")
    print(f"{'-'*t_w}-+-{'-'*s_w}-+-{'-'*c_w}-+-{'-'*p_w}-+-{'-'*tp_w}-+-{'-'*sl_w}-+-{'-'*rr_w}")
    
    for p in plans:
        sig = p.direction
        sig_disp = f"🟢 {sig}" if sig == "BUY" else f"🔴 {sig}" if sig == "SELL" else f"⚪ {sig}"
            
        print(
            f"{p.ticker:<{t_w}} | "
            f"{sig_disp:<{s_w}} | "
            f"{p.confidence:<{c_w}.2f} | "
            f"{p.entry_price:<{p_w}.2f} | "
            f"{p.take_profit:<{tp_w}.2f} | "
            f"{p.stop_loss:<{sl_w}.2f} | "
            f"{p.risk_reward:<{rr_w}.2f}"
        )
        
    print(f"{'-'*(t_w+s_w+c_w+p_w+tp_w+sl_w+rr_w + 18)}")
    
    # Extract only actionable signals to print detailed views
    active_plans = [p for p in plans if p.direction != "WAIT"]
    
    if active_plans:
        print(f"\n\n🔥 DETAILED ACTIVE TRADE PLANS ({len(active_plans)}) 🔥")
        print(f"{'='*85}\n")
        for plan in active_plans:
            print(plan.to_markdown())
            print(f"\n{'-'*40}\n")
    else:
        print("\n😴 No active signals found today. Market is quiet.")
        

def main():
    parser = argparse.ArgumentParser(description="Agent-Monkey - Morning Multi-Asset Scanner")
    parser.add_argument(
        "--tickers", 
        nargs='+', 
        default=WATCHLIST, 
        help="List of tickers to scan (e.g. --tickers BTC/USDT ETH/USDT)"
    )
    parser.add_argument("--period", type=str, default="6mo", help="Data period to fetch (default: 6mo)")
    parser.add_argument("--interval", type=str, default="1d", help="Candle interval (default: 1d)")
    
    args = parser.parse_args()
    
    plans = scan_market(args.tickers, args.period, args.interval)
    if plans:
        display_summary(plans)
    else:
        print("\n❌ Scanner finished but no assets were processed successfully.")

if __name__ == "__main__":
    main()
