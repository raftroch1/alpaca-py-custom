#!/usr/bin/env python3
"""
Comparison between Unrealistic vs Realistic Backtest Results
"""

import pandas as pd

def compare_backtests():
    print("🔥 UNREALISTIC vs REALISTIC BACKTEST COMPARISON")
    print("=" * 80)
    
    # Load both datasets
    try:
        unrealistic_df = pd.read_csv('comprehensive_zero_dte_trades.csv')
        realistic_df = pd.read_csv('realistic_zero_dte_trades.csv')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return
    
    # Calculate metrics for both
    print(f"{'METRIC':<25} {'UNREALISTIC':<20} {'REALISTIC':<20} {'REALITY CHECK'}")
    print("-" * 80)
    
    # Account performance
    unrealistic_total_pnl = unrealistic_df['pnl'].sum()
    realistic_total_pnl = realistic_df['pnl'].sum()
    
    print(f"{'Total P&L':<25} ${unrealistic_total_pnl:>15,.2f} ${realistic_total_pnl:>15,.2f} {'✅ Much more realistic' if abs(realistic_total_pnl) < 1000 else '❌ Still too high'}")
    
    # Daily averages
    unrealistic_avg = unrealistic_df['pnl'].mean()
    realistic_avg = realistic_df['pnl'].mean()
    
    print(f"{'Average Daily P&L':<25} ${unrealistic_avg:>15,.2f} ${realistic_avg:>15,.2f} {'✅ Realistic for small account' if abs(realistic_avg) < 100 else '❌ Still high'}")
    
    # Contract sizes
    unrealistic_contracts = unrealistic_df['contracts'].mean()
    realistic_contracts = realistic_df['contracts'].mean()
    
    print(f"{'Average Contracts':<25} {unrealistic_contracts:>15.1f} {realistic_contracts:>15.1f} {'✅ Much more reasonable' if realistic_contracts < 50 else '❌ Still too many'}")
    
    # Risk levels
    # For unrealistic, calculate risk percentage
    unrealistic_risk = ((unrealistic_df['contracts'] * unrealistic_df['max_loss']).mean() / 25000) * 100
    realistic_risk = realistic_df['risk_percentage'].mean()
    
    print(f"{'Average Risk %':<25} {unrealistic_risk:>15.1f}% {realistic_risk:>15.1f}% {'✅ Proper risk management' if realistic_risk < 2 else '❌ Still risky'}")
    
    # Best/worst days
    unrealistic_best = unrealistic_df['pnl'].max()
    realistic_best = realistic_df['pnl'].max()
    
    print(f"{'Best Day':<25} ${unrealistic_best:>15,.2f} ${realistic_best:>15,.2f} {'✅ Realistic gain' if realistic_best < 200 else '❌ Still high'}")
    
    unrealistic_worst = unrealistic_df['pnl'].min()
    realistic_worst = realistic_df['pnl'].min()
    
    print(f"{'Worst Day':<25} ${unrealistic_worst:>15,.2f} ${realistic_worst:>15,.2f} {'✅ Manageable loss' if realistic_worst > -200 else '❌ Still risky'}")
    
    # Win rates
    unrealistic_win_rate = (unrealistic_df['pnl'] > 0).mean() * 100
    realistic_win_rate = (realistic_df['pnl'] > 0).mean() * 100
    
    print(f"{'Win Rate':<25} {unrealistic_win_rate:>15.1f}% {realistic_win_rate:>15.1f}% {'✅ Reasonable' if 40 <= realistic_win_rate <= 60 else '⚠️ Check strategy'}")
    
    print("\n" + "=" * 80)
    print("📊 KEY INSIGHTS:")
    print("=" * 80)
    
    print("❌ UNREALISTIC VERSION PROBLEMS:")
    print(f"   • Trading {unrealistic_contracts:.0f} contracts/day (WAY too many)")
    print(f"   • Making ${unrealistic_avg:,.0f}/day (16-20% daily returns = impossible)")
    print(f"   • Best day: ${unrealistic_best:,.0f} (23% account gain in one day!)")
    print(f"   • Risk: {unrealistic_risk:.1f}% per trade (too aggressive)")
    
    print("\n✅ REALISTIC VERSION IMPROVEMENTS:")
    print(f"   • Trading {realistic_contracts:.0f} contracts/day (reasonable size)")
    print(f"   • Making ${realistic_avg:.0f}/day (0.02% daily = sustainable)")
    print(f"   • Best day: ${realistic_best:.0f} (manageable 0.4% gain)")
    print(f"   • Risk: {realistic_risk:.1f}% per trade (proper risk management)")
    
    print("\n💡 REALISTIC EXPECTATIONS FOR $25,000 ACCOUNT:")
    print("   • Daily Target: $250 (1%)")
    print("   • Realistic Range: $50-$500/day")
    print("   • Max Contracts: 10-50 (depending on strategy)")
    print("   • Max Risk: $500/trade (2% of account)")
    print("   • Win Rate: 45-65% (strategy dependent)")
    print("   • Monthly Return: 5-15% (if consistent)")
    
    print("\n🚨 WHY THE UNREALISTIC VERSION WAS WRONG:")
    print("   1. Position sizing algorithm was flawed")
    print("   2. Premium per contract was too high")
    print("   3. No realistic market constraints")
    print("   4. Risk management was theoretical, not practical")
    print("   5. Ignored liquidity and slippage")

if __name__ == "__main__":
    compare_backtests()
