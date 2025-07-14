#!/usr/bin/env python3
"""
Detailed Daily Analysis for 0DTE Backtest

This script provides granular analysis of each trading day to understand:
- Position sizing decisions
- Risk per trade
- Premium collection/payment details
- Account impact analysis
- Realistic vs unrealistic trades
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DailyTradeAnalyzer:
    """Analyze each trading day in detail."""
    
    def __init__(self, trades_csv: str = "comprehensive_zero_dte_trades.csv"):
        """Initialize with trades data."""
        try:
            self.trades_df = pd.read_csv(trades_csv)
            print(f"✅ Loaded {len(self.trades_df)} trades from {trades_csv}")
        except FileNotFoundError:
            print(f"❌ Could not find {trades_csv}")
            self.trades_df = pd.DataFrame()
    
    def analyze_position_sizing_logic(self):
        """Analyze the position sizing logic for each trade."""
        print("=" * 80)
        print("🔍 DETAILED POSITION SIZING ANALYSIS")
        print("=" * 80)
        
        if self.trades_df.empty:
            print("No trades to analyze")
            return
        
        # Account parameters (matching the backtest)
        initial_account = 25000.0
        max_risk_per_trade = 0.02  # 2%
        
        print(f"�� POSITION SIZING RULES:")
        print(f"   Max Risk Per Trade: {max_risk_per_trade:.1%}")
        print(f"   Initial Account: ${initial_account:,.2f}")
        print(f"   Max Risk Amount Per Trade: ${initial_account * max_risk_per_trade:,.2f}")
        print()
        
        for idx, trade in self.trades_df.iterrows():
            account_value = trade['account_value'] - trade['pnl']  # Account before this trade
            max_risk_amount = account_value * max_risk_per_trade
            
            # Calculate realistic contract size
            max_loss_per_contract = trade['max_loss']
            realistic_contracts = int(max_risk_amount / max_loss_per_contract) if max_loss_per_contract > 0 else 1
            realistic_contracts = max(1, realistic_contracts)
            
            # Actual vs realistic comparison
            actual_contracts = trade['contracts']
            actual_risk = actual_contracts * max_loss_per_contract
            actual_risk_pct = (actual_risk / account_value) * 100
            
            print(f"📅 {trade['date']} - {trade['strategy'].upper()}")
            print(f"   Account Value: ${account_value:,.2f}")
            print(f"   VIX: {trade['vix']:.2f} | SPY: ${trade['spy_price']:.2f}")
            print(f"   Max Loss Per Contract: ${max_loss_per_contract:.2f}")
            print(f"   Realistic Contracts: {realistic_contracts}")
            print(f"   ACTUAL Contracts: {actual_contracts}")
            print(f"   Actual Risk: ${actual_risk:,.2f} ({actual_risk_pct:.1f}% of account)")
            print(f"   P&L per Contract: ${trade['pnl_per_contract']:.2f}")
            print(f"   Total P&L: ${trade['pnl']:,.2f}")
            
            # Risk analysis
            if actual_risk_pct > 5:
                print(f"   ⚠️  WARNING: High risk trade ({actual_risk_pct:.1f}%)")
            if actual_contracts > realistic_contracts * 2:
                print(f"   🚨 ALERT: Over-leveraged by {actual_contracts/realistic_contracts:.1f}x")
            
            # Premium analysis
            premium_collected = actual_contracts * abs(trade['pnl_per_contract'])
            print(f"   Premium: ${premium_collected:,.2f}")
            print(f"   Return on Risk: {(trade['pnl'] / actual_risk * 100) if actual_risk > 0 else 0:.1f}%")
            print()
    
    def analyze_daily_performance(self):
        """Analyze daily performance patterns."""
        print("=" * 80)
        print("📈 DAILY PERFORMANCE ANALYSIS")
        print("=" * 80)
        
        # Group by date to see if multiple trades per day
        daily_summary = self.trades_df.groupby('date').agg({
            'pnl': ['count', 'sum', 'mean'],
            'contracts': 'sum',
            'strategy': lambda x: ', '.join(x.unique()),
            'account_value': 'last'
        }).round(2)
        
        daily_summary.columns = ['Trades_Count', 'Total_PnL', 'Avg_PnL', 'Total_Contracts', 'Strategies', 'End_Account_Value']
        
        print(f"📊 TRADES PER DAY SUMMARY:")
        print(daily_summary)
        print()
        
        # Check for multiple trades per day
        multiple_trade_days = daily_summary[daily_summary['Trades_Count'] > 1]
        if not multiple_trade_days.empty:
            print(f"🔍 DAYS WITH MULTIPLE TRADES:")
            print(multiple_trade_days)
        else:
            print("✅ Only one trade per day (as expected)")
        print()
    
    def analyze_risk_vs_reward(self):
        """Analyze risk vs reward for each strategy."""
        print("=" * 80)
        print("⚖️  RISK vs REWARD ANALYSIS BY STRATEGY")
        print("=" * 80)
        
        strategies = self.trades_df['strategy'].unique()
        
        for strategy in strategies:
            strategy_trades = self.trades_df[self.trades_df['strategy'] == strategy]
            
            print(f"🎯 {strategy.upper()} ANALYSIS:")
            print(f"   Number of Trades: {len(strategy_trades)}")
            print(f"   Win Rate: {(strategy_trades['pnl'] > 0).mean():.1%}")
            print(f"   Average P&L: ${strategy_trades['pnl'].mean():,.2f}")
            print(f"   Average Contracts: {strategy_trades['contracts'].mean():.1f}")
            print(f"   Average P&L per Contract: ${strategy_trades['pnl_per_contract'].mean():.2f}")
            print(f"   Best Trade: ${strategy_trades['pnl'].max():,.2f}")
            print(f"   Worst Trade: ${strategy_trades['pnl'].min():,.2f}")
            
            # Risk analysis
            avg_max_loss = strategy_trades['max_loss'].mean()
            avg_contracts = strategy_trades['contracts'].mean()
            avg_risk_per_trade = avg_max_loss * avg_contracts
            
            print(f"   Average Max Loss per Contract: ${avg_max_loss:.2f}")
            print(f"   Average Risk per Trade: ${avg_risk_per_trade:,.2f}")
            print()
    
    def identify_unrealistic_trades(self):
        """Identify potentially unrealistic trades."""
        print("=" * 80)
        print("🚨 UNREALISTIC TRADE ANALYSIS")
        print("=" * 80)
        
        # Define thresholds for unrealistic trades
        high_profit_threshold = 1000  # > $1000 profit
        high_risk_threshold = 0.05   # > 5% of account
        
        unrealistic_count = 0
        
        for idx, trade in self.trades_df.iterrows():
            account_before = trade['account_value'] - trade['pnl']
            risk_pct = (trade['contracts'] * trade['max_loss']) / account_before
            
            is_unrealistic = False
            warnings = []
            
            # Check for high profits
            if trade['pnl'] > high_profit_threshold:
                warnings.append(f"High profit: ${trade['pnl']:,.2f}")
                is_unrealistic = True
            
            # Check for high risk
            if risk_pct > high_risk_threshold:
                warnings.append(f"High risk: {risk_pct:.1%} of account")
                is_unrealistic = True
            
            # Check for excessive contracts
            if trade['contracts'] > 100:
                warnings.append(f"Many contracts: {trade['contracts']}")
                is_unrealistic = True
            
            if is_unrealistic:
                unrealistic_count += 1
                print(f"🚨 {trade['date']} - {trade['strategy'].upper()}")
                print(f"   Account: ${account_before:,.2f}")
                print(f"   Contracts: {trade['contracts']}")
                print(f"   P&L: ${trade['pnl']:,.2f}")
                print(f"   Warnings: {', '.join(warnings)}")
                print()
        
        print(f"📊 SUMMARY: {unrealistic_count}/{len(self.trades_df)} trades flagged as unrealistic")
    
    def suggest_realistic_position_sizing(self):
        """Suggest more realistic position sizing."""
        print("=" * 80)
        print("💡 REALISTIC POSITION SIZING SUGGESTIONS")
        print("=" * 80)
        
        initial_account = 25000.0
        conservative_risk = 0.01  # 1% risk per trade
        moderate_risk = 0.02      # 2% risk per trade
        
        print(f"📊 SUGGESTED POSITION SIZING RULES:")
        print(f"   Conservative Risk: {conservative_risk:.1%} per trade")
        print(f"   Moderate Risk: {moderate_risk:.1%} per trade")
        print(f"   Account Size: ${initial_account:,.2f}")
        print()
        
        # Calculate realistic trades
        print(f"🎯 REALISTIC TRADE EXAMPLES:")
        
        # Example Iron Condor
        ic_max_loss = 5.0  # $5 max loss per contract
        conservative_contracts = int((initial_account * conservative_risk) / ic_max_loss)
        moderate_contracts = int((initial_account * moderate_risk) / ic_max_loss)
        
        print(f"   Iron Condor (${ic_max_loss:.2f} max loss/contract):")
        print(f"     Conservative: {conservative_contracts} contracts (${conservative_contracts * ic_max_loss:,.2f} risk)")
        print(f"     Moderate: {moderate_contracts} contracts (${moderate_contracts * ic_max_loss:,.2f} risk)")
        
        # Example Diagonal Spread
        ds_max_loss = 3.0  # $3 max loss per contract
        conservative_contracts_ds = int((initial_account * conservative_risk) / ds_max_loss)
        moderate_contracts_ds = int((initial_account * moderate_risk) / ds_max_loss)
        
        print(f"   Diagonal Spread (${ds_max_loss:.2f} max loss/contract):")
        print(f"     Conservative: {conservative_contracts_ds} contracts (${conservative_contracts_ds * ds_max_loss:,.2f} risk)")
        print(f"     Moderate: {moderate_contracts_ds} contracts (${moderate_contracts_ds * ds_max_loss:,.2f} risk)")
        print()
        
        print(f"💰 EXPECTED DAILY RETURNS:")
        print(f"   Target: 1% daily = ${initial_account * 0.01:.2f}")
        print(f"   Conservative trades should target: ${initial_account * 0.005:.2f} - ${initial_account * 0.01:.2f}")
        print(f"   Moderate trades should target: ${initial_account * 0.01:.2f} - ${initial_account * 0.02:.2f}")
    
    def run_full_analysis(self):
        """Run complete analysis."""
        print("🔍 COMPREHENSIVE 0DTE TRADE ANALYSIS")
        print("=" * 80)
        
        if self.trades_df.empty:
            print("❌ No trade data available for analysis")
            return
        
        # 1. Position sizing analysis
        self.analyze_position_sizing_logic()
        
        # 2. Daily performance
        self.analyze_daily_performance()
        
        # 3. Risk vs reward by strategy
        self.analyze_risk_vs_reward()
        
        # 4. Identify unrealistic trades
        self.identify_unrealistic_trades()
        
        # 5. Suggest realistic sizing
        self.suggest_realistic_position_sizing()

def main():
    """Main analysis function."""
    analyzer = DailyTradeAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
