#!/usr/bin/env python3
"""
TRUE HIGH FREQUENCY 0DTE STRATEGY V2 - BACKTEST - PROFIT OPTIMIZED
Target: $250-$500 daily profit on $25K account (1-2% daily returns)

This backtest:
- Tests the V2 PROFIT OPTIMIZED strategy 
- Enhanced win rate optimization (targeting 55%+ vs v1's 39.3%)
- Dynamic position sizing and adaptive profit targets
- Comprehensive profit analysis and tracking
- Follows all established development rules

Version: v2 - PROFIT OPTIMIZED
Author: Strategy Development Framework
Date: 2025-01-17
"""

import sys
import os

# Add strategies path to import the strategy
strategies_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'strategies')
sys.path.append(strategies_path)

try:
    from high_frequency_0dte_v2 import TrueHighFrequency0DTEStrategyV2
except ImportError as e:
    print(f"❌ Error importing V2 strategy: {e}")
    print("📁 Strategies path:", strategies_path)
    print("🔍 Available files:", os.listdir(strategies_path) if os.path.exists(strategies_path) else "Path not found")
    sys.exit(1)

import pandas as pd
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def setup_logging():
    """Setup enhanced logging for V2 backtest"""
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'strategies', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'high_frequency_0dte_v2_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def save_detailed_results(strategy, results, start_date, end_date):
    """Save comprehensive V2 results with profit analysis"""
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'strategies', 'logs')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Summary CSV
    summary_file = os.path.join(results_dir, f'high_frequency_0dte_v2_summary_{timestamp}.csv')
    summary_df = pd.DataFrame([{
        'strategy_version': 'v2',
        'backtest_period': f"{start_date}_to_{end_date}",
        'total_trades': results['total_trades'],
        'daily_trade_frequency': results['daily_trade_frequency'],
        'win_rate_pct': results['win_rate'],
        'total_pnl': results['total_pnl'],
        'daily_pnl_avg': results['daily_pnl_avg'],
        'total_return_pct': results['total_return_pct'],
        'final_capital': results['final_capital'],
        'target_achieved': results['target_achieved'],
        'daily_target_hit_rate': results['daily_target_hit_rate'],
        'high_conviction_trades': results['high_conviction_trades'],
        'momentum_trades': results['momentum_trades'],
        'profitable_days': results['profitable_days'],
        'profitable_days_pct': results['profitable_days_pct'],
        'max_daily_profit': results['max_daily_profit'],
        'min_daily_profit': results['min_daily_profit'],
        'regime_trending': results['regime_breakdown'].get('trending', 0),
        'regime_ranging': results['regime_breakdown'].get('ranging', 0),
        'regime_volatile': results['regime_breakdown'].get('volatile', 0)
    }])
    summary_df.to_csv(summary_file, index=False)
    
    # 2. Daily Performance CSV
    daily_file = os.path.join(results_dir, f'high_frequency_0dte_v2_daily_{timestamp}.csv')
    if results['daily_profits']:
        daily_df = pd.DataFrame({
            'day_number': range(1, len(results['daily_profits']) + 1),
            'daily_pnl': results['daily_profits'],
            'target_hit': [pnl >= 250 for pnl in results['daily_profits']],
            'cumulative_pnl': pd.Series(results['daily_profits']).cumsum()
        })
        daily_df.to_csv(daily_file, index=False)
    
    # 3. Detailed Trades CSV
    if hasattr(strategy, 'trades') and strategy.trades:
        trades_file = os.path.join(results_dir, f'high_frequency_0dte_v2_trades_{timestamp}.csv')
        trades_df = pd.DataFrame(strategy.trades)
        trades_df.to_csv(trades_file, index=False)
        
        print(f"📊 Saved detailed trade log: {trades_file}")
    
    print(f"📈 Results saved:")
    print(f"   Summary: {summary_file}")
    print(f"   Daily Performance: {daily_file}")
    
    return summary_file, daily_file

def analyze_v2_improvements(results):
    """Analyze V2 improvements and provide recommendations"""
    
    print("\n" + "="*70)
    print("🔍 V2 PROFIT OPTIMIZATION ANALYSIS")
    print("="*70)
    
    # Profitability Analysis
    daily_target_low = 250
    daily_target_high = 500
    actual_daily = results['daily_pnl_avg']
    
    print(f"💰 PROFIT ANALYSIS:")
    print(f"   Target Range: ${daily_target_low} - ${daily_target_high} per day")
    print(f"   Actual Average: ${actual_daily:+.2f} per day")
    
    if actual_daily >= daily_target_low:
        print(f"   ✅ SUCCESS: Target achieved!")
        if actual_daily >= daily_target_high:
            print(f"   🎉 EXCELLENT: Exceeded high target by ${actual_daily - daily_target_high:+.2f}")
    else:
        gap = daily_target_low - actual_daily
        print(f"   ⚠️  Gap to target: ${gap:+.2f} per day")
        print(f"   📊 Progress: {(actual_daily/daily_target_low)*100:.1f}% of minimum target")
    
    # Win Rate Analysis
    print(f"\n🏆 WIN RATE ANALYSIS:")
    print(f"   Current: {results['win_rate']:.1f}%")
    if results['win_rate'] >= 55:
        print(f"   ✅ EXCELLENT: Win rate above 55% target")
    elif results['win_rate'] >= 50:
        print(f"   ✅ GOOD: Win rate above 50%")
    elif results['win_rate'] >= 45:
        print(f"   ⚠️  ACCEPTABLE: Room for improvement")
    else:
        print(f"   ❌ NEEDS WORK: Win rate below 45%")
    
    # Trade Frequency Analysis
    print(f"\n⚡ FREQUENCY ANALYSIS:")
    print(f"   Daily Trades: {results['daily_trade_frequency']:.1f}")
    if results['daily_trade_frequency'] >= 20:
        print(f"   ✅ HIGH FREQUENCY: Excellent trade generation")
    elif results['daily_trade_frequency'] >= 15:
        print(f"   ✅ GOOD: Above minimum 15 trades/day")
    else:
        print(f"   ⚠️  LOW: Below 15 trades/day target")
    
    # Quality Analysis
    high_conv_rate = results['high_conviction_trades'] / results['total_trades'] * 100 if results['total_trades'] > 0 else 0
    momentum_rate = results['momentum_trades'] / results['total_trades'] * 100 if results['total_trades'] > 0 else 0
    
    print(f"\n🎯 SIGNAL QUALITY ANALYSIS:")
    print(f"   High Conviction Trades: {results['high_conviction_trades']} ({high_conv_rate:.1f}%)")
    print(f"   Momentum Trades: {results['momentum_trades']} ({momentum_rate:.1f}%)")
    print(f"   Daily Target Hit Rate: {results['daily_target_hit_rate']:.1f}%")
    print(f"   Profitable Days: {results['profitable_days_pct']:.1f}%")
    
    # Recommendations
    print(f"\n💡 V3 OPTIMIZATION RECOMMENDATIONS:")
    
    if actual_daily < daily_target_low:
        print(f"   🔧 Increase average profit per trade")
        print(f"   🔧 Optimize position sizing for winners")
        print(f"   🔧 Improve profit target adaptation")
    
    if results['win_rate'] < 55:
        print(f"   🔧 Enhance signal filtering quality")
        print(f"   🔧 Add market timing filters")
        print(f"   🔧 Improve entry timing precision")
    
    if results['daily_target_hit_rate'] < 80:
        print(f"   🔧 Focus on consistency improvements")
        print(f"   🔧 Add daily profit tracking stops")
        print(f"   🔧 Implement adaptive daily targets")
    
    print("="*70)

def main():
    """Run V2 PROFIT OPTIMIZED backtest"""
    
    # Setup logging
    logger = setup_logging()
    
    print("🚀 TRUE HIGH FREQUENCY 0DTE STRATEGY V2 - PROFIT OPTIMIZED BACKTEST")
    print("="*80)
    print("🎯 Target: $250-$500 daily profit on $25K account")
    print("📊 Focus: Enhanced win rate and profit optimization")
    print("🔧 Improvements: Dynamic sizing, adaptive targets, quality filtering")
    print("="*80)
    
    # Initialize V2 strategy
    try:
        strategy = TrueHighFrequency0DTEStrategyV2()
        logger.info("✅ V2 Strategy initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize V2 strategy: {e}")
        return
    
    # Backtest parameters
    start_date = "20241208"  # December 8, 2024
    end_date = "20241213"    # December 13, 2024 (6 trading days)
    
    logger.info(f"📅 Testing period: {start_date} to {end_date}")
    logger.info(f"💰 Starting capital: ${strategy.starting_capital:,.2f}")
    
    # Run the backtest
    try:
        print("\n🔄 Running V2 PROFIT OPTIMIZED backtest...")
        results = strategy.run_backtest(start_date, end_date)
        
        if results['total_trades'] == 0:
            logger.error("❌ No trades generated - check strategy parameters")
            return
            
        # Save comprehensive results
        print("\n💾 Saving V2 results...")
        summary_file, daily_file = save_detailed_results(strategy, results, start_date, end_date)
        
        # Analyze V2 improvements
        analyze_v2_improvements(results)
        
        # Final summary
        print(f"\n🎉 V2 BACKTEST COMPLETE!")
        print(f"📊 Generated {results['total_trades']} trades over {results['trading_days']} days")
        print(f"💰 Final P&L: ${results['total_pnl']:+,.2f}")
        print(f"📈 Daily Average: ${results['daily_pnl_avg']:+,.2f}")
        print(f"🏆 Win Rate: {results['win_rate']:.1f}%")
        print(f"⚡ Trade Frequency: {results['daily_trade_frequency']:.1f} trades/day")
        
        # Target achievement status
        if results['target_achieved']:
            print(f"🎯 ✅ DAILY TARGET ACHIEVED: ${results['daily_pnl_avg']:+.2f} >= $250")
        else:
            needed = 250 - results['daily_pnl_avg']
            print(f"🎯 ⚠️ Need ${needed:+.2f} more daily profit to hit minimum target")
        
        logger.info("✅ V2 Backtest completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 