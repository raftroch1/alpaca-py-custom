#!/usr/bin/env python3
"""
Cached Strategy Runner - Fast Strategy Testing with Cached Data
==============================================================

This runner loads cached ThetaData and runs strategy logic without API calls.
Perfect for iterating on strategy parameters quickly.

Usage:
    python cached_strategy_runner.py --start_date 20250701 --end_date 20250717
    python cached_strategy_runner.py --date 20250717  # Single day
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import gzip
import argparse
from typing import Dict, List, Optional
import time

# Import the ThetaData collector
from thetadata_collector import ThetaDataCollector

class CachedStrategyRunner:
    """Run strategies using cached ThetaData for fast iteration"""
    
    def __init__(self, cache_dir: str = "cached_data"):
        self.cache_dir = cache_dir
        self.collector = ThetaDataCollector(cache_dir)
        
        # Strategy parameters (from V2-REAL)
        self.confidence_threshold = 0.55  # Higher confidence for better win rate
        self.max_daily_trades = 15
        self.min_option_price = 0.50
        self.max_option_price = 3.00
        
        # Performance tracking
        self.trades = []
        self.daily_pnl = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        print(f"🎯 Strategy: True High Frequency 0DTE V2-CACHED")
        print(f"📁 Cache directory: {self.cache_dir}")
    
    def load_cached_data(self, date: str) -> Dict:
        """Load all cached data for a specific date"""
        # Load SPY minute bars
        spy_bars = self.collector.load_from_cache("spy_bars", date)
        
        # Load option chain
        option_chain = self.collector.load_from_cache("option_chains", date)
        
        if spy_bars is None:
            print(f"❌ No SPY data cached for {date}")
            return {}
        
        if option_chain is None:
            print(f"❌ No option chain cached for {date}")
            return {}
        
        return {
            'spy_bars': spy_bars,
            'option_chain': option_chain,
            'date': date
        }
    
    def calculate_signals(self, spy_bars: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading signals from SPY minute bars
        (Simplified version of V2-REAL strategy logic)
        """
        # Calculate technical indicators
        spy_bars = spy_bars.copy()
        
        # Ensure unique index to avoid pandas errors
        if spy_bars.index.duplicated().any():
            spy_bars = spy_bars.reset_index().drop_duplicates().set_index('datetime')
        
        # RSI
        delta = spy_bars['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        spy_bars['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        spy_bars['sma_5'] = spy_bars['close'].rolling(window=5).mean()
        spy_bars['sma_20'] = spy_bars['close'].rolling(window=20).mean()
        
        # Volume analysis
        spy_bars['volume_sma'] = spy_bars['volume'].rolling(window=20).mean()
        spy_bars['volume_ratio'] = spy_bars['volume'] / spy_bars['volume_sma']
        
        # Price momentum
        spy_bars['price_change'] = spy_bars['close'].pct_change(periods=5)
        spy_bars['momentum'] = spy_bars['close'].rolling(window=10).mean().pct_change(periods=3)
        
        # Initialize signal columns
        spy_bars['signal'] = 0
        spy_bars['confidence'] = 0.0
        
        # Call signals (bullish)
        call_conditions = (
            (spy_bars['rsi'] < 40) &  # Oversold
            (spy_bars['close'] > spy_bars['sma_5']) &  # Above short MA
            (spy_bars['volume_ratio'] > 1.2) &  # High volume
            (spy_bars['momentum'] > 0.001)  # Positive momentum
        )
        
        # Put signals (bearish)
        put_conditions = (
            (spy_bars['rsi'] > 60) &  # Overbought
            (spy_bars['close'] < spy_bars['sma_5']) &  # Below short MA
            (spy_bars['volume_ratio'] > 1.2) &  # High volume
            (spy_bars['momentum'] < -0.001)  # Negative momentum
        )
        
        # Set signals using iloc to avoid index issues
        call_indices = spy_bars.index[call_conditions]
        put_indices = spy_bars.index[put_conditions]
        
        spy_bars.loc[call_indices, 'signal'] = 1  # Call signal
        spy_bars.loc[put_indices, 'signal'] = -1  # Put signal
        
        # Calculate confidence for signal rows only
        signal_mask = spy_bars['signal'] != 0
        if signal_mask.any():
            confidence_values = (
                abs(spy_bars.loc[signal_mask, 'momentum']) * 100 +
                spy_bars.loc[signal_mask, 'volume_ratio'] * 0.2 +
                abs(50 - spy_bars.loc[signal_mask, 'rsi']) * 0.01
            )
            spy_bars.loc[signal_mask, 'confidence'] = confidence_values
        
        return spy_bars
    
    def find_best_option(self, option_chain: Dict, signal: int, spy_price: float) -> Dict:
        """
        Find best option to trade based on signal
        
        Args:
            option_chain: Cached option chain data
            signal: 1 for calls, -1 for puts
            spy_price: Current SPY price
            
        Returns:
            Dict with option details or empty dict if none found
        """
        if signal == 1:  # Call signal
            options = option_chain.get('calls', {})
            target_strike = spy_price + 2  # Slightly OTM call
        else:  # Put signal
            options = option_chain.get('puts', {})
            target_strike = spy_price - 2  # Slightly OTM put
        
        # Find closest strike to target
        best_strike = None
        best_price = None
        min_distance = float('inf')
        
        for strike, price in options.items():
            if self.min_option_price <= price <= self.max_option_price:
                distance = abs(strike - target_strike)
                if distance < min_distance:
                    min_distance = distance
                    best_strike = strike
                    best_price = price
        
        if best_strike is None:
            return {}
        
        return {
            'strike': best_strike,
            'price': best_price,
            'type': 'call' if signal == 1 else 'put',
            'signal': signal
        }
    
    def simulate_trade(self, entry_time: str, option: Dict, spy_bars: pd.DataFrame) -> Dict:
        """
        Simulate a trade execution and exit
        
        Args:
            entry_time: Entry timestamp
            option: Option details
            spy_bars: SPY minute bars for the day
            
        Returns:
            Trade result
        """
        entry_price = option['price']
        
        # Find bars after entry time
        entry_idx = spy_bars.index.get_loc(entry_time) if entry_time in spy_bars.index else 0
        remaining_bars = spy_bars.iloc[entry_idx + 1:]
        
        if len(remaining_bars) == 0:
            return {}  # No time left to trade
        
        # Simple exit logic: take profit at 20% or stop loss at -50%
        # (In reality, we'd use real option pricing models)
        profit_target = entry_price * 1.20
        stop_loss = entry_price * 0.50
        
        # Simulate option price movement based on SPY movement
        spy_entry_price = spy_bars.loc[entry_time, 'close']
        
        for timestamp, row in remaining_bars.iterrows():
            spy_current_price = row['close']
            spy_move_pct = (spy_current_price - spy_entry_price) / spy_entry_price
            
            # Rough option price simulation (delta approximation)
            if option['type'] == 'call':
                option_price_estimate = entry_price * (1 + spy_move_pct * 5)  # 5x leverage approximation
            else:
                option_price_estimate = entry_price * (1 - spy_move_pct * 5)
            
            # Check exit conditions
            if option_price_estimate >= profit_target:
                exit_price = profit_target
                exit_reason = "profit_target"
                break
            elif option_price_estimate <= stop_loss:
                exit_price = stop_loss
                exit_reason = "stop_loss"
                break
        else:
            # End of day exit
            exit_price = option_price_estimate
            exit_reason = "eod"
        
        pnl = exit_price - entry_price
        return {
            'entry_time': entry_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'option': option
        }
    
    def run_strategy_day(self, date: str) -> Dict:
        """Run strategy for a single day using cached data"""
        print(f"\n📅 Running strategy for {date}")
        
        # Load cached data
        data = self.load_cached_data(date)
        if not data:
            return {'trades': [], 'pnl': 0}
        
        spy_bars = data['spy_bars']
        option_chain = data['option_chain']
        
        print(f"📊 Loaded {len(spy_bars)} minute bars")
        print(f"🔍 SPY price range: ${spy_bars['close'].min():.2f} - ${spy_bars['close'].max():.2f}")
        
        # Calculate signals
        spy_with_signals = self.calculate_signals(spy_bars)
        
        # Diagnostic: Count signals generated
        call_signals = (spy_with_signals['signal'] == 1).sum()
        put_signals = (spy_with_signals['signal'] == -1).sum()
        high_confidence = (spy_with_signals['confidence'] >= self.confidence_threshold).sum()
        
        print(f"🎯 Signals generated: {call_signals} calls, {put_signals} puts")
        print(f"⚡ High confidence signals (>= {self.confidence_threshold}): {high_confidence}")
        print(f"📋 Available options: {len(option_chain.get('calls', {}))} calls, {len(option_chain.get('puts', {}))} puts")
        
        # Find trading opportunities
        day_trades = []
        daily_pnl = 0
        trade_count = 0
        signals_checked = 0
        
        # Process every 5th minute to reduce noise (like V2-REAL optimization)
        for i in range(0, len(spy_with_signals), 5):
            if trade_count >= self.max_daily_trades:
                break
                
            row = spy_with_signals.iloc[i]
            
            # Check for signal above confidence threshold
            if abs(row['signal']) == 1 and row['confidence'] >= self.confidence_threshold:
                signals_checked += 1
                
                # Diagnostic output for first few signals
                if signals_checked <= 3:
                    signal_type = "CALL" if row['signal'] == 1 else "PUT"
                    print(f"  🔍 Signal {signals_checked}: {signal_type} at {row.name.strftime('%H:%M')} "
                          f"(confidence: {row['confidence']:.2f}, SPY: ${row['close']:.2f})")
                
                # Find best option
                option = self.find_best_option(option_chain, row['signal'], row['close'])
                
                if option:
                    # Simulate trade
                    trade = self.simulate_trade(row.name, option, spy_with_signals)
                    
                    if trade:
                        day_trades.append(trade)
                        daily_pnl += trade['pnl']
                        trade_count += 1
                        
                        print(f"  🔥 Trade {trade_count}: {option['type'].upper()} ${option['strike']} "
                              f"| Entry: ${trade['entry_price']:.2f} Exit: ${trade['exit_price']:.2f} "
                              f"| P&L: ${trade['pnl']:.2f} ({trade['exit_reason']})")
                elif signals_checked <= 3:
                    print(f"    ❌ No valid options found for this signal")
        
        if signals_checked > 3:
            print(f"  ... (checked {signals_checked} total qualifying signals)")
        
        print(f"📊 Day Summary: {len(day_trades)} trades, ${daily_pnl:.2f} P&L")
        
        return {
            'date': date,
            'trades': day_trades,
            'pnl': daily_pnl,
            'trade_count': len(day_trades),
            'signals_generated': call_signals + put_signals,
            'high_confidence_signals': high_confidence
        }
    
    def run_backtest(self, start_date: str, end_date: str) -> Dict:
        """Run full backtest using cached data"""
        print("🚀 CACHED STRATEGY BACKTEST")
        print("=" * 60)
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"🎯 Confidence threshold: {self.confidence_threshold}")
        print(f"📈 Max daily trades: {self.max_daily_trades}")
        print(f"💰 Option price range: ${self.min_option_price} - ${self.max_option_price}")
        
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        all_results = []
        total_pnl = 0
        total_trades = 0
        winning_trades = 0
        
        current_date = start_dt
        while current_date <= end_dt:
            if current_date.weekday() < 5:  # Trading days only
                date_str = current_date.strftime('%Y%m%d')
                
                # Run strategy for this day
                day_result = self.run_strategy_day(date_str)
                
                if day_result['trades']:
                    all_results.append(day_result)
                    total_pnl += day_result['pnl']
                    total_trades += day_result['trade_count']
                    
                    # Count winning trades
                    day_winners = sum(1 for trade in day_result['trades'] if trade['pnl'] > 0)
                    winning_trades += day_winners
            
            current_date += timedelta(days=1)
        
        # Calculate final metrics
        trading_days = len(all_results)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_daily_pnl = total_pnl / trading_days if trading_days > 0 else 0
        avg_trades_per_day = total_trades / trading_days if trading_days > 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 FINAL BACKTEST RESULTS")
        print("=" * 60)
        print(f"📅 Trading days: {trading_days}")
        print(f"⚡ Total trades: {total_trades}")
        print(f"📈 Trades per day: {avg_trades_per_day:.1f}")
        print(f"🎯 Win rate: {win_rate:.1f}%")
        print(f"💰 Total P&L: ${total_pnl:.2f}")
        print(f"📊 Avg daily P&L: ${avg_daily_pnl:.2f}")
        
        # Performance evaluation
        if avg_daily_pnl >= 250:
            print("✅ TARGET ACHIEVED: Meeting $250-500 daily profit goal!")
        elif avg_daily_pnl >= 150:
            print("⚠️ CLOSE TO TARGET: Within range, needs optimization")
        else:
            print("❌ BELOW TARGET: Requires strategy improvements")
        
        return {
            'trading_days': trading_days,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_daily_pnl': avg_daily_pnl,
            'avg_trades_per_day': avg_trades_per_day,
            'daily_results': all_results
        }


def main():
    parser = argparse.ArgumentParser(description="Cached Strategy Runner - Fast strategy testing")
    parser.add_argument('--start_date', help='Start date (YYYYMMDD)')
    parser.add_argument('--end_date', help='End date (YYYYMMDD)')
    parser.add_argument('--date', help='Single date (YYYYMMDD)')
    parser.add_argument('--confidence', type=float, default=0.55, help='Confidence threshold')
    parser.add_argument('--max_trades', type=int, default=15, help='Max trades per day')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = CachedStrategyRunner()
    
    # Set strategy parameters if provided
    if args.confidence:
        runner.confidence_threshold = args.confidence
    if args.max_trades:
        runner.max_daily_trades = args.max_trades
    
    # Run backtest
    if args.date:
        # Single day
        result = runner.run_strategy_day(args.date)
        print(f"\nSingle day result: {result}")
    elif args.start_date and args.end_date:
        # Date range
        results = runner.run_backtest(args.start_date, args.end_date)
    else:
        # Default: recent 5 trading days
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
        print(f"📅 Using default date range: {start_date} to {end_date}")
        results = runner.run_backtest(start_date, end_date)


if __name__ == "__main__":
    main() 