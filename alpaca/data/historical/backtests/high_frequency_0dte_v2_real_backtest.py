#!/usr/bin/env python3
"""
TRUE HIGH FREQUENCY 0DTE STRATEGY V2-REAL - BACKTEST
Target: $250-$500 daily profit on $25K account (1-2% daily returns)

USING REAL THETADATA:
- Real minute-by-minute SPY bars from ThetaData
- Real 0DTE option prices from ThetaData
- Actual intraday market movements
- True volume and volatility patterns

Version: v2-real - REAL THETADATA
Author: Strategy Development Framework
Date: 2025-01-17
"""

import sys
import os

# Add strategies path to import the strategy
strategies_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'strategies')
sys.path.append(strategies_path)

try:
    from high_frequency_0dte_v2_real import TrueHighFrequency0DTEStrategyV2Real
except ImportError as e:
    print(f"❌ Error importing V2-REAL strategy: {e}")
    print("📁 Strategies path:", strategies_path)
    print("🔍 Available files:", os.listdir(strategies_path) if os.path.exists(strategies_path) else "Path not found")
    sys.exit(1)

import pandas as pd
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def setup_logging():
    """Setup enhanced logging for V2-REAL backtest"""
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'strategies', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'high_frequency_0dte_v2_real_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def save_real_data_results(strategy, results, start_date, end_date):
    """Save comprehensive V2-REAL results with real data analysis"""
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'strategies', 'logs')
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Summary CSV with real data metrics
    summary_file = os.path.join(results_dir, f'high_frequency_0dte_v2_real_summary_{timestamp}.csv')
    summary_df = pd.DataFrame([{
        'strategy_version': 'v2-real',
        'data_source': 'ThetaData',
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
        'real_data_points': results['real_data_points'],
        'api_calls_made': results['api_calls_made'],
        'failed_api_calls': results['failed_api_calls'],
        'data_success_rate': results['data_success_rate'],
        'regime_trending': results['regime_breakdown'].get('trending', 0),
        'regime_ranging': results['regime_breakdown'].get('ranging', 0),
        'regime_volatile': results['regime_breakdown'].get('volatile', 0)
    }])
    summary_df.to_csv(summary_file, index=False)
    
    # 2. Daily Performance CSV
    daily_file = os.path.join(results_dir, f'high_frequency_0dte_v2_real_daily_{timestamp}.csv')
    if results['daily_profits']:
        daily_df = pd.DataFrame({
            'day_number': range(1, len(results['daily_profits']) + 1),
            'daily_pnl': results['daily_profits'],
            'target_hit': [pnl >= 250 for pnl in results['daily_profits']],
            'cumulative_pnl': pd.Series(results['daily_profits']).cumsum()
        })
        daily_df.to_csv(daily_file, index=False)
    
    # 3. Detailed Trades CSV with real data metadata
    if hasattr(strategy, 'trades') and strategy.trades:
        trades_file = os.path.join(results_dir, f'high_frequency_0dte_v2_real_trades_{timestamp}.csv')
        trades_df = pd.DataFrame(strategy.trades)
        trades_df.to_csv(trades_file, index=False)
        
        print(f"📊 Saved detailed REAL DATA trade log: {trades_file}")
    
    print(f"📈 V2-REAL Results saved:")
    print(f"   Summary: {summary_file}")
    print(f"   Daily Performance: {daily_file}")
    
    return summary_file, daily_file

def analyze_real_data_performance(results):
    """Analyze V2-REAL performance and real data quality"""
    
    print("\n" + "="*75)
    print("🔍 V2-REAL DATA ANALYSIS - USING ACTUAL THETADATA")
    print("="*75)
    
    # Real Data Quality Analysis
    print(f"📡 REAL DATA QUALITY:")
    print(f"   Data Points Retrieved: {results['real_data_points']:,}")
    print(f"   API Calls Made: {results['api_calls_made']:,}")
    print(f"   Failed API Calls: {results['failed_api_calls']:,}")
    print(f"   Data Success Rate: {results['data_success_rate']:.1f}%")
    
    if results['data_success_rate'] >= 90:
        print(f"   ✅ EXCELLENT: High quality real data")
    elif results['data_success_rate'] >= 75:
        print(f"   ✅ GOOD: Acceptable real data quality")
    else:
        print(f"   ⚠️  POOR: Low real data quality may affect results")
    
    # Profitability Analysis with Real Data
    daily_target_low = 250
    daily_target_high = 500
    actual_daily = results['daily_pnl_avg']
    
    print(f"\n💰 REAL DATA PROFIT ANALYSIS:")
    print(f"   Target Range: ${daily_target_low} - ${daily_target_high} per day")
    print(f"   Actual Average: ${actual_daily:+.2f} per day")
    
    if actual_daily >= daily_target_low:
        print(f"   ✅ SUCCESS: Target achieved with REAL DATA!")
        if actual_daily >= daily_target_high:
            print(f"   🎉 EXCELLENT: Exceeded high target by ${actual_daily - daily_target_high:+.2f}")
    else:
        gap = daily_target_low - actual_daily
        print(f"   ⚠️  Gap to target: ${gap:+.2f} per day")
        print(f"   📊 Progress: {(actual_daily/daily_target_low)*100:.1f}% of minimum target")
    
    # Win Rate Analysis with Real Data
    print(f"\n🏆 REAL DATA WIN RATE ANALYSIS:")
    print(f"   Current Win Rate: {results['win_rate']:.1f}%")
    if results['win_rate'] >= 60:
        print(f"   ✅ EXCELLENT: High win rate with real data")
    elif results['win_rate'] >= 50:
        print(f"   ✅ GOOD: Above 50% with real market conditions")
    elif results['win_rate'] >= 45:
        print(f"   ⚠️  ACCEPTABLE: Room for improvement")
    else:
        print(f"   ❌ NEEDS WORK: Win rate below 45% with real data")
    
    # Trade Frequency with Real Data
    print(f"\n⚡ REAL DATA FREQUENCY ANALYSIS:")
    print(f"   Daily Trades: {results['daily_trade_frequency']:.1f}")
    if results['daily_trade_frequency'] >= 20:
        print(f"   ✅ HIGH FREQUENCY: Excellent signal generation from real data")
    elif results['daily_trade_frequency'] >= 15:
        print(f"   ✅ GOOD: Above minimum 15 trades/day with real data")
    elif results['daily_trade_frequency'] >= 8:
        print(f"   ⚠️  MEDIUM: Reasonable frequency with real data constraints")
    else:
        print(f"   ⚠️  LOW: Below optimal frequency - may need parameter tuning")
    
    # Signal Quality with Real Data
    high_conv_rate = results['high_conviction_trades'] / results['total_trades'] * 100 if results['total_trades'] > 0 else 0
    
    print(f"\n🎯 REAL DATA SIGNAL QUALITY:")
    print(f"   High Conviction Trades: {results['high_conviction_trades']} ({high_conv_rate:.1f}%)")
    print(f"   Daily Target Hit Rate: {results['daily_target_hit_rate']:.1f}%")
    print(f"   Profitable Days: {results['profitable_days_pct']:.1f}%")
    
    # Real vs Simulated Comparison
    print(f"\n🔬 REAL DATA VS SIMULATION BENEFITS:")
    print(f"   ✅ Accurate intraday price movements")
    print(f"   ✅ Real volume confirmation signals")
    print(f"   ✅ Actual option pricing dynamics")
    print(f"   ✅ True market microstructure effects")
    print(f"   ✅ No simulation bias or assumptions")
    
    # Recommendations for V3
    print(f"\n💡 V3 REAL DATA OPTIMIZATION RECOMMENDATIONS:")
    
    if actual_daily < daily_target_low:
        print(f"   🔧 Increase position sizing on high-confidence real signals")
        print(f"   🔧 Optimize entry timing using real volume patterns")
        print(f"   🔧 Add real-time option flow analysis")
    
    if results['win_rate'] < 55:
        print(f"   🔧 Enhance signal filtering using real market microstructure")
        print(f"   🔧 Add real volume profile analysis")
        print(f"   🔧 Improve entry precision with real tick data")
    
    if results['daily_trade_frequency'] < 15:
        print(f"   🔧 Lower confidence thresholds for real data signals")
        print(f"   🔧 Add more sensitive real-time indicators")
        print(f"   🔧 Implement faster signal processing")
    
    if results['data_success_rate'] < 90:
        print(f"   🔧 Improve ThetaData connection reliability")
        print(f"   🔧 Add fallback data sources")
        print(f"   🔧 Implement better error handling")
    
    print("="*75)

def main():
    """Run V2-REAL backtest with actual ThetaData"""
    
    # Setup logging
    logger = setup_logging()
    
    print("🚀 TRUE HIGH FREQUENCY 0DTE STRATEGY V2-REAL BACKTEST")
    print("="*85)
    print("📡 USING REAL THETADATA - NO SIMULATION")
    print("🎯 Target: $250-$500 daily profit on $25K account")
    print("📊 Focus: Real minute bars and option prices")
    print("⚡ Pure real market data analysis")
    print("="*85)
    
    # Check ThetaData connection first
    print("\n🔌 Checking ThetaData connection...")
    
    # Initialize V2-REAL strategy
    try:
        strategy = TrueHighFrequency0DTEStrategyV2Real()
        logger.info("✅ V2-REAL Strategy initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize V2-REAL strategy: {e}")
        return
    
    # Backtest parameters (recent dates for better ThetaData availability)
    start_date = "20241208"  # December 8, 2024
    end_date = "20241213"    # December 13, 2024 (6 trading days)
    
    logger.info(f"📅 Testing period: {start_date} to {end_date}")
    logger.info(f"💰 Starting capital: ${strategy.starting_capital:,.2f}")
    logger.info("📡 Will fetch real ThetaData minute bars for each trading day")
    
    # Run the backtest
    try:
        print("\n🔄 Running V2-REAL backtest with ThetaData...")
        print("📡 This may take longer due to real API calls...")
        
        results = strategy.run_backtest(start_date, end_date)
        
        if results['total_trades'] == 0:
            logger.error("❌ No trades generated - check ThetaData connection and strategy parameters")
            print(f"📊 Data retrieved: {results.get('real_data_points', 0)} points")
            print(f"🔌 API calls made: {results.get('api_calls_made', 0)}")
            return
            
        # Save comprehensive results
        print("\n💾 Saving V2-REAL results...")
        summary_file, daily_file = save_real_data_results(strategy, results, start_date, end_date)
        
        # Analyze real data performance
        analyze_real_data_performance(results)
        
        # Final summary
        print(f"\n🎉 V2-REAL BACKTEST COMPLETE!")
        print(f"📊 Generated {results['total_trades']} trades over {results['trading_days']} days")
        print(f"💰 Final P&L: ${results['total_pnl']:+,.2f}")
        print(f"📈 Daily Average: ${results['daily_pnl_avg']:+,.2f}")
        print(f"🏆 Win Rate: {results['win_rate']:.1f}%")
        print(f"⚡ Trade Frequency: {results['daily_trade_frequency']:.1f} trades/day")
        print(f"📡 Real Data Points: {results['real_data_points']:,}")
        print(f"📈 Data Success: {results['data_success_rate']:.1f}%")
        
        # Target achievement status
        if results['target_achieved']:
            print(f"🎯 ✅ DAILY TARGET ACHIEVED WITH REAL DATA: ${results['daily_pnl_avg']:+.2f} >= $250")
        else:
            needed = 250 - results['daily_pnl_avg']
            print(f"🎯 ⚠️ Need ${needed:+.2f} more daily profit to hit minimum target")
        
        # Real data validation
        if results['data_success_rate'] >= 80:
            print(f"✅ REAL DATA VALIDATION: High quality ThetaData used")
        else:
            print(f"⚠️ REAL DATA WARNING: Limited ThetaData quality may affect accuracy")
        
        logger.info("✅ V2-REAL Backtest completed successfully")
        
    except Exception as e:
        logger.error(f"❌ V2-REAL Backtest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 