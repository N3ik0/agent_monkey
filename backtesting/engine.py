import pandas as pd
import numpy as np
from typing import List, Dict, Any

class BacktestEngine:
    """
    Simulates trading execution over a chronological list of MAS signals.
    Enforces risk management rules (position sizing, SL/TP) and prevents 
    look-ahead bias by resolving trades based on subsequent OHLCV data.
    """

    def __init__(self, initial_capital: float = 1000.0, risk_per_trade: float = 0.02):
        """
        Args:
            initial_capital (float): Starting capital for the backtest.
            risk_per_trade (float): Fraction of capital to risk per trade (e.g., 0.02 for 2%).
        """
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade

    def run(self, processed_df: pd.DataFrame, signals: List[dict], ticker: str) -> Dict[str, Any]:
        """
        Executes the backtest simulation.

        Args:
            processed_df (pd.DataFrame): Market data containing OHLCV and technical features.
            signals (List[dict]): List of consensus signals from the orchestrator.
            ticker (str): The asset being tested.

        Returns:
            dict: Performance metrics and detailed trade history.
        """
        capital = self.initial_capital
        peak_capital = capital
        max_drawdown = 0.0
        
        trades = []
        is_in_position = False
        current_trade = None
        
        if not signals:
            return self._empty_result(ticker)
            
        # Map signals by Date for O(1) chronological lookup during iteration
        signal_map = {s["Date"]: s for s in signals}
        first_signal_date = signals[0]["Date"]
        
        # Attach readable string dates matching the signal formatting
        df_dates = pd.Series([str(x.date()) if hasattr(x, 'date') else str(x) for x in processed_df.index], index=processed_df.index)
        
        # Determine the df index where simulation begins
        start_indices = np.where(df_dates == first_signal_date)[0]
        if len(start_indices) == 0:
            return self._empty_result(ticker)
            
        start_idx = start_indices[0]
        
        pending_entry = None
        for i in range(start_idx, len(processed_df)):
            row = processed_df.iloc[i]
            date_str = df_dates.iloc[i]
            
            # a. Activate pending_entry if exists (= enter position at start of this candle)
            if pending_entry is not None and not is_in_position:
                current_trade = pending_entry
                is_in_position = True
                pending_entry = None

            # b. Update existing position (Check TP / SL execution)
            if is_in_position:
                high = float(row['high'])
                low = float(row['low'])
                
                sl_hit = False
                tp_hit = False
                
                if current_trade['direction'] == 'BUY':
                    if low <= current_trade['sl']: sl_hit = True
                    if high >= current_trade['tp']: tp_hit = True
                else: # SELL
                    if high >= current_trade['sl']: sl_hit = True
                    if low <= current_trade['tp']: tp_hit = True
                    
                if sl_hit or tp_hit:
                    # Resolve Trade
                    if sl_hit and tp_hit:
                        # Pessimistic execution: assume SL is hit first if both triggered in same candle
                        exit_price = current_trade['sl']
                        outcome = 'LOSS'
                    elif sl_hit:
                        exit_price = current_trade['sl']
                        outcome = 'LOSS'
                    else:
                        exit_price = current_trade['tp']
                        outcome = 'WIN'
                        
                    if current_trade['direction'] == 'BUY':
                        pnl = (exit_price - current_trade['entry']) * current_trade['position_size']
                    else:
                        pnl = (current_trade['entry'] - exit_price) * current_trade['position_size']
                        
                    capital += pnl
                    if capital > peak_capital:
                        peak_capital = capital
                        
                    dd = (peak_capital - capital) / peak_capital
                    if dd > max_drawdown:
                        max_drawdown = dd
                        
                    current_trade['exit_date'] = date_str
                    current_trade['exit_price'] = exit_price
                    current_trade['pnl'] = pnl
                    current_trade['outcome'] = outcome
                    trades.append(current_trade)
                    is_in_position = False
                    current_trade = None

            # c. Check for new signals if no position is active
            if not is_in_position and pending_entry is None and date_str in signal_map:
                sig_info = signal_map[date_str]
                signal = sig_info["Signal"]
                
                if signal in ["BUY", "SELL"]:
                    entry = float(row['close'])
                    raw_atr = row.get("ATR_14", pd.NA)
                    atr = float(raw_atr) if not pd.isna(raw_atr) else pd.NA
                    
                    if not pd.isna(atr) and atr > 0:
                        # Hardcoded risk profile parameters (1.5x ATR SL, 1:2 R:R)
                        atr_multiplier = 1.5
                        rr_ratio = 2.0
                        
                        if signal == "BUY":
                            sl = entry - (atr * atr_multiplier)
                            tp = entry + (atr * atr_multiplier * rr_ratio)
                            risk_per_unit = entry - sl
                        else:
                            sl = entry + (atr * atr_multiplier)
                            tp = entry - (atr * atr_multiplier * rr_ratio)
                            risk_per_unit = sl - entry
                            
                        if risk_per_unit > 0:
                            # Position Sizing based on fixed risk fraction
                            money_at_risk = capital * self.risk_per_trade
                            position_size = money_at_risk / risk_per_unit
                            
                            pending_entry = {
                                'ticker': ticker,
                                'entry_date': date_str,
                                'direction': signal,
                                'entry': entry,
                                'sl': sl,
                                'tp': tp,
                                'position_size': position_size,
                                'confidence': sig_info.get('Confiance', 0.0)
                            }

        nb_trades = len(trades)
        wins = [t for t in trades if t['outcome'] == 'WIN']
        win_rate = len(wins) / nb_trades if nb_trades > 0 else 0.0
        
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0.0)
        
        total_return = (capital - self.initial_capital) / self.initial_capital
        
        return {
            'ticker': ticker,
            'initial_capital': self.initial_capital,
            'capital_final': capital,
            'total_return': total_return,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'nb_trades': nb_trades,
            'trades': trades
        }
        
    def _empty_result(self, ticker: str) -> dict:
        return {
            'ticker': ticker,
            'initial_capital': self.initial_capital,
            'capital_final': self.initial_capital,
            'total_return': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'profit_factor': 0.0,
            'nb_trades': 0,
            'trades': []
        }
        
    def print_report(self, result: Dict[str, Any]):
        """
        Outputs a beautifully formatted terminal report of the simulated strategy.
        """
        print(f"\n{'='*70}")
        print(f"💰 BACKTEST ENGINE RESULT: {result['ticker']}")
        print(f"{'='*70}")
        
        print(f"Initial Capital:  {result['initial_capital']:.2f}")
        print(f"Final Capital:    {result['capital_final']:.2f}")
        
        # Color coding return
        ret = result['total_return'] * 100
        ret_colored = f"\033[92m{ret:.2f}%\033[0m" if ret > 0 else f"\033[91m{ret:.2f}%\033[0m" if ret < 0 else f"{ret:.2f}%"
        print(f"Total Return:     {ret_colored}")
        
        print(f"Win Rate:         {result['win_rate']*100:.2f}%")
        print(f"Max Drawdown:     \033[91m{result['max_drawdown']*100:.2f}%\033[0m")
        print(f"Profit Factor:    {result['profit_factor']:.2f}")
        print(f"Total Trades:     {result['nb_trades']}")
        
        if result['nb_trades'] > 0:
            print(f"\n📝 DETAILED TRADES HISTORY")
            print(f"{'-'*96}")
            print(f"{'Entry Date':<12} | {'Exit Date':<12} | {'Dir':<4} | {'Entry':<10} | {'Exit':<10} | {'Size':<8} | {'PnL':<8} | {'Outcome'}")
            print(f"{'-'*12}-+-{'-'*12}-+-{'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}")
            
            for t in result['trades']:
                pnl = t['pnl']
                pnl_str = f"{pnl:.2f}"
                pnl_disp = f"\033[92m+{pnl_str}\033[0m" if pnl > 0 else f"\033[91m{pnl_str}\033[0m"
                
                dir_color = "🟢 BUY " if t['direction'] == "BUY" else "🔴 SELL"
                outcome_color = "✅ WIN " if t['outcome'] == "WIN" else "❌ LOSS"
                
                print(
                    f"{t['entry_date']:<12} | "
                    f"{t['exit_date']:<12} | "
                    f"{dir_color:<4} | "
                    f"{t['entry']:<10.2f} | "
                    f"{t['exit_price']:<10.2f} | "
                    f"{t['position_size']:<8.4f} | "
                    f"{pnl_disp:<17} | " # 17 because ANSI codes add invisible chars
                    f"{outcome_color}"
                )
            print(f"{'-'*96}\n")
