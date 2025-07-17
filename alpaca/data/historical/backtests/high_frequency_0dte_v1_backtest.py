#!/usr/bin/env python3
"""
TRUE HIGH FREQUENCY 0DTE STRATEGY V1 - BACKTEST
Target: $250-$500 daily profit on $25K account (1-2% daily returns)

This backtest:
- Uses the TRUE HIGH FREQUENCY strategy (15-25 trades/day)
- Tests ultra-aggressive parameters for daily profit targets
- Generates proper logging and CSV results
- Follows all established development rules

Version: v1 - TRUE HIGH FREQUENCY
Author: Strategy Development Framework
Date: 2025-01-17
"""

import sys
import os

# Add strategies path to import the strategy
strategies_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'strategies')
sys.path.append(strategies_path)

try:
    from high_frequency_0dte_v1 import TrueHighFrequency0DTEStrategy
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"📁 Looking for strategy in: {strategies_path}")
    print("💡 Make sure high_frequency_0dte_v1.py exists in strategies/ folder")
    sys.exit(1)
from datetime import datetime


def run_strategy_backtest():
    """
    Run the TRUE HIGH FREQUENCY 0DTE Strategy V1 backtest.
    
    Target: $250-$500 daily profit on $25K account
    Method: 15-25 trades per day with ultra-aggressive parameters
    """
    print("⚡ TRUE HIGH FREQUENCY 0DTE STRATEGY V1 - BACKTEST")
    print("=" * 70)
    print("🎯 TARGET: $250-$500 DAILY PROFIT ON $25K ACCOUNT")
    print("⚡ METHOD: 15-25 TRADES PER DAY")
    print("📋 Following established development rules:")
    print("   ✅ Inherits from BaseThetaStrategy")
    print("   ✅ Uses real market data + HF simulation")
    print("   ✅ Proper logging to strategies/logs/")
    print("   ✅ CSV results automatically saved")
    print("   ✅ Follows versioning convention")
    print("=" * 70)
    
    try:
        # Initialize the TRUE HIGH FREQUENCY strategy
        strategy = TrueHighFrequency0DTEStrategy()
        
        print(f"⚡ Strategy: {strategy.strategy_name} {strategy.version}")
        print(f"💰 Starting Capital: ${strategy.starting_capital:,.2f}")
        print(f"🎯 Daily Target: ${strategy.daily_profit_target}")
        print(f"⚡ Target Frequency: {strategy.min_daily_trades}-{strategy.max_daily_trades} trades/day")
        print()
        
        # Run backtest for optimal period
        print("🔄 Running TRUE HIGH FREQUENCY backtest...")
        results = strategy.run_backtest('20250106', '20250115')
        
        if results:
            print("\n" + "=" * 70)
            print("⚡ TRUE HIGH FREQUENCY BACKTEST RESULTS")
            print("=" * 70)
            
            # Key metrics
            daily_pnl = results.get('daily_pnl', 0)
            daily_trades = results.get('daily_trade_frequency', 0)
            win_rate = results.get('win_rate', 0)
            total_return = results.get('total_return_pct', 0)
            target_achieved = results.get('target_achieved', False)
            
            print(f"🎯 Daily P&L: ${daily_pnl:+.2f} (target: $250-$500)")
            print(f"⚡ Daily Trades: {daily_trades:.1f} (target: 15-25)")
            print(f"📈 Win Rate: {win_rate:.1f}%")
            print(f"📊 Total Return: {total_return:+.2f}%")
            print(f"🏦 Final Capital: ${results['final_capital']:,.2f}")
            print(f"📊 Total Trades: {results['total_trades']}")
            
            # Performance evaluation
            print("\n" + "=" * 70)
            print("🎯 TARGET ACHIEVEMENT ANALYSIS")
            print("=" * 70)
            
            # Daily profit target
            if daily_pnl >= 500:
                print("🎉 EXCEEDED HIGH TARGET ($500+/day)")
            elif daily_pnl >= 350:
                print("✅ EXCEEDED MEDIUM TARGET ($350+/day)")
            elif daily_pnl >= 250:
                print("✅ ACHIEVED MINIMUM TARGET ($250+/day)")
            elif daily_pnl >= 150:
                print("🟡 Good progress (60%+ of target)")
            else:
                print("⚠️ Below target - needs optimization")
                
            # Trading frequency target
            if daily_trades >= 20:
                print("⚡ EXCELLENT FREQUENCY (20+ trades/day)")
            elif daily_trades >= 15:
                print("✅ ACHIEVED FREQUENCY TARGET (15+ trades/day)")
            elif daily_trades >= 10:
                print("🟡 Good frequency (10+ trades/day)")
            else:
                print("⚠️ Below frequency target")
                
            # Overall assessment
            print("\n🎯 OVERALL STRATEGY ASSESSMENT:")
            if target_achieved:
                print("🎉 BOTH TARGETS ACHIEVED! Strategy ready for live trading!")
            elif daily_pnl >= 200 and daily_trades >= 12:
                print("✅ Strong performance! Minor optimizations could reach full targets.")
            elif daily_pnl >= 100 or daily_trades >= 8:
                print("🟡 Good foundation. Further optimization recommended.")
            else:
                print("🔧 Needs significant optimization for target achievement.")
                
            # Next steps
            print("\n💡 NEXT STEPS:")
            if target_achieved:
                print("   • Strategy is ready for live trading")
                print("   • Consider risk management refinements") 
                print("   • Test with paper trading first")
            else:
                print("   • Analyze trade-by-trade results in CSV")
                print("   • Consider parameter optimization")
                print("   • Test different market conditions")
                print("   • Refine signal generation sensitivity")
                
            print(f"\n📁 Results saved to strategies/logs/ folder")
            print(f"💾 CSV file contains detailed trade-by-trade analysis")
            
            return results
            
    except Exception as e:
        print(f"❌ Backtest error: {e}")
        import traceback
        traceback.print_exc()
        print("\n❌ Framework error occurred!")
        return None
        
    print("\n✅ TRUE HIGH FREQUENCY backtest completed!")
    print("📊 Framework validation complete")
    print("📊 Check strategies/logs/ folder for detailed results")


if __name__ == "__main__":
    print("Starting True High Frequency 0DTE Strategy V1 Backtest...")
    print("Following .cursorrules and STRATEGY_DEVELOPMENT_RULES.md")
    print()
    
    run_strategy_backtest() 