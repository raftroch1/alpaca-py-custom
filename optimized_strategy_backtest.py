#!/usr/bin/env python3
"""
Optimized Strategy Backtest - Using Best Parameters
==================================================

Runs a comprehensive backtest using the optimal parameters discovered
by the strategy optimizer.

Usage:
    python optimized_strategy_backtest.py
"""

import os
from datetime import datetime
from strategy_optimizer import AdvancedStrategyOptimizer

def run_optimized_backtest():
    """Run backtest with optimized parameters"""
    
    print("🏆 OPTIMIZED STRATEGY BACKTEST")
    print("=" * 60)
    print("📊 Using best parameters discovered by optimizer")
    print("🎯 Target: Beat $8.79 daily P&L benchmark")
    
    # Best parameters from optimization
    optimal_params = {
        # Core signal parameters (from optimization)
        'confidence_threshold': 0.25,
        'min_signal_score': 4,
        'bull_momentum_threshold': 0.0015,
        'bear_momentum_threshold': 0.0015,
        'volume_threshold': 1.8,
        'momentum_weight': 3,
        'max_daily_trades': 10,
        'sample_frequency': 5,
        
        # Technical indicator parameters
        'ema_fast': 8,
        'ema_slow': 21,
        'sma_period': 20,
        'rsi_oversold': 45,
        'rsi_overbought': 55,
        'technical_weight': 2,
        'volume_weight': 2,
        'pattern_weight': 1,
        'min_volume_ratio': 1.2,
        
        # Option strategy parameters
        'call_strike_offset': 1.5,
        'put_strike_offset': 1.5,
        'call_base_price': 1.40,
        'put_base_price': 1.30
    }
    
    # Available dates
    dates = ['20250106', '20250107', '20250108', '20250110', '20250113', '20250114', '20250115']
    
    # Initialize optimizer with correct cache directory
    optimizer = AdvancedStrategyOptimizer(cache_dir="alpaca/data/historical/strategies/cached_data")
    
    # Load cached data
    print(f"\n📅 Loading cached data for {len(dates)} trading days...")
    all_data = optimizer.load_cached_data(dates)
    
    if not all_data:
        print("❌ No cached data found!")
        return
    
    # Run backtest with optimal parameters
    start_time = datetime.now()
    result = optimizer.test_parameter_set(all_data, optimal_params)
    backtest_time = (datetime.now() - start_time).total_seconds()
    
    if not result:
        print("❌ Backtest failed!")
        return
    
    print(f"\n🚀 OPTIMIZED STRATEGY RESULTS")
    print("=" * 60)
    print(f"⚡ Backtest completed in {backtest_time:.2f} seconds")
    print(f"📅 Trading period: {len(dates)} days")
    print(f"📊 Data processed: {sum(len(data) for data in all_data.values()):,} minute bars")
    
    # Overall performance
    print(f"\n💰 PERFORMANCE METRICS:")
    print(f"   📈 Total P&L: ${result['total_pnl']:.2f}")
    print(f"   📊 Avg Daily P&L: ${result['avg_daily_pnl']:.2f}")
    print(f"   🎯 Win Rate: {result['win_rate']:.1f}%")
    print(f"   ⚡ Total Trades: {result['total_trades']}")
    print(f"   📈 Avg Trades/Day: {result['avg_trades_per_day']:.1f}")
    
    # Performance evaluation against target
    target_daily_pnl = 250  # $250-500 target range
    
    if result['avg_daily_pnl'] >= 500:
        status = "🟢 EXCEEDS TARGET"
        print(f"\n✅ {status}: Significantly above $250-500 range!")
    elif result['avg_daily_pnl'] >= 250:
        status = "🟢 TARGET ACHIEVED"
        print(f"\n✅ {status}: Within $250-500 daily profit goal!")
    elif result['avg_daily_pnl'] >= 150:
        status = "🟡 CLOSE TO TARGET"
        print(f"\n⚠️ {status}: Getting close, needs minor optimization")
    else:
        status = "🔴 BELOW TARGET"
        print(f"\n❌ {status}: Requires further optimization")
    
    # Daily breakdown
    print(f"\n📅 DAILY PERFORMANCE BREAKDOWN:")
    print("-" * 50)
    
    for daily in result['daily_results']:
        profit_emoji = "📈" if daily['pnl'] > 0 else "📉"
        print(f"   {daily['date']}: {daily['trades']} trades | "
              f"${daily['pnl']:.2f} P&L | {daily['win_rate']:.1f}% WR {profit_emoji}")
    
    # Trade analysis
    all_trades = result['all_trades']
    profitable_trades = [t for t in all_trades if t['pnl'] > 0]
    losing_trades = [t for t in all_trades if t['pnl'] <= 0]
    
    if profitable_trades:
        avg_win = sum(t['pnl'] for t in profitable_trades) / len(profitable_trades)
        print(f"\n📊 TRADE ANALYSIS:")
        print(f"   💚 Winning trades: {len(profitable_trades)} (avg: ${avg_win:.2f})")
        
    if losing_trades:
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades)
        print(f"   🔴 Losing trades: {len(losing_trades)} (avg: ${avg_loss:.2f})")
    
    # Best trades
    best_trades = sorted(all_trades, key=lambda x: x['pnl'], reverse=True)[:3]
    print(f"\n🏆 BEST TRADES:")
    for i, trade in enumerate(best_trades, 1):
        print(f"   {i}. {trade['time']} {trade['type']} ${trade['strike']} | "
              f"${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} | "
              f"${trade['pnl']:.2f} ({trade['outcome']})")
    
    # Parameter summary
    print(f"\n⚙️ OPTIMAL PARAMETERS USED:")
    key_params = [
        'confidence_threshold', 'min_signal_score', 'bull_momentum_threshold',
        'volume_threshold', 'max_daily_trades'
    ]
    for param in key_params:
        if param in optimal_params:
            print(f"   • {param}: {optimal_params[param]}")
    
    # System efficiency
    print(f"\n⚡ SYSTEM EFFICIENCY:")
    print(f"   🚀 Backtest speed: {backtest_time:.2f} seconds")
    print(f"   📊 Bars per second: {sum(len(data) for data in all_data.values()) / backtest_time:,.0f}")
    print(f"   💾 Using cached data: Instant vs 10+ min API calls")
    
    # Future recommendations
    print(f"\n🔮 NEXT STEPS:")
    if result['avg_daily_pnl'] >= 250:
        print(f"   ✅ Strategy ready for live testing")
        print(f"   🎯 Consider paper trading with these parameters")
        print(f"   📈 Monitor performance on new market data")
    else:
        print(f"   🔧 Continue parameter optimization")
        print(f"   📊 Test different time periods")
        print(f"   🎯 Refine entry/exit logic")
    
    return result


def main():
    result = run_optimized_backtest()
    
    if result and result['avg_daily_pnl'] >= 250:
        print(f"\n🎉 CONGRATULATIONS!")
        print(f"🏆 Optimized strategy achieves target daily profit goal!")
        print(f"💡 This represents a major breakthrough in strategy development!")
    else:
        print(f"\n📈 PROGRESS MADE!")
        print(f"🔧 Continue optimization to reach target performance!")


if __name__ == "__main__":
    main() 