#!/usr/bin/env python3
"""
REALISTIC 0DTE Options Strategy Backtest

This version implements realistic:
- Position sizing (5-50 contracts max)
- Premium amounts ($50-300 per trade)
- Risk management (1-2% max risk)
- Profit targets ($50-500 daily)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealisticOptionsStrategy:
    """Realistic options strategy with proper position sizing."""
    
    def __init__(self):
        self.commission = 0.65  # Per contract commission
        
    def iron_condor(self, spot_price: float, strike_width: float = 5.0) -> dict:
        """Realistic Iron Condor with proper premium amounts."""
        # Realistic premium for iron condor
        net_credit_per_contract = np.random.uniform(1.50, 3.50)  # $1.50-3.50 credit
        max_loss_per_contract = strike_width - net_credit_per_contract
        
        return {
            'strategy': 'iron_condor',
            'net_credit_per_contract': net_credit_per_contract,
            'max_loss_per_contract': max_loss_per_contract,
            'max_profit_per_contract': net_credit_per_contract - (4 * self.commission),
            'profit_prob': 0.65  # 65% profit probability
        }
    
    def iron_butterfly(self, spot_price: float, wing_width: float = 10.0) -> dict:
        """Realistic Iron Butterfly with proper premium amounts."""
        # Realistic premium for iron butterfly
        net_credit_per_contract = np.random.uniform(2.00, 4.00)  # $2-4 credit
        max_loss_per_contract = wing_width - net_credit_per_contract
        
        return {
            'strategy': 'iron_butterfly',
            'net_credit_per_contract': net_credit_per_contract,
            'max_loss_per_contract': max_loss_per_contract,
            'max_profit_per_contract': net_credit_per_contract - (4 * self.commission),
            'profit_prob': 0.55  # 55% profit probability
        }
    
    def diagonal_spread(self, spot_price: float) -> dict:
        """Realistic Diagonal Spread with proper debit amounts."""
        # Realistic debit for diagonal spread
        net_debit_per_contract = np.random.uniform(1.00, 2.50)  # $1-2.50 debit
        max_profit_per_contract = np.random.uniform(2.00, 5.00)  # $2-5 profit potential
        
        return {
            'strategy': 'diagonal_spread',
            'net_debit_per_contract': net_debit_per_contract,
            'max_loss_per_contract': net_debit_per_contract + (2 * self.commission),
            'max_profit_per_contract': max_profit_per_contract,
            'profit_prob': 0.45  # 45% profit probability (directional play)
        }

class RealisticBacktester:
    """Realistic backtesting with proper risk management."""
    
    def __init__(self):
        self.strategy = RealisticOptionsStrategy()
        self.trades = []
        
        # Realistic account parameters
        self.initial_account_value = 25000.0
        self.current_account_value = self.initial_account_value
        self.daily_profit_target = 250.0  # $250/day (1%)
        self.max_risk_per_trade = 500.0   # $500 max risk per trade
        self.max_daily_loss = 750.0       # $750 max daily loss
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.current_date = None
        
        # Strategy parameters
        self.vix_low = 17
        self.vix_high = 18
        
    def calculate_realistic_position_size(self, trade_setup: dict) -> int:
        """Calculate realistic position size."""
        max_loss_per_contract = trade_setup['max_loss_per_contract']
        
        # Calculate contracts based on max risk
        max_contracts = int(self.max_risk_per_trade / max_loss_per_contract)
        
        # Realistic constraints
        min_contracts = 1
        max_realistic_contracts = min(50, max_contracts)  # Never more than 50 contracts
        
        # For smaller accounts, limit further
        if self.current_account_value < 30000:
            max_realistic_contracts = min(30, max_realistic_contracts)
        
        contracts = max(min_contracts, min(max_realistic_contracts, max_contracts))
        
        return contracts
    
    def simulate_realistic_outcome(self, trade_setup: dict, contracts: int) -> float:
        """Simulate realistic trade outcome."""
        profit_prob = trade_setup['profit_prob']
        
        if np.random.random() < profit_prob:
            # Profitable trade - partial profit
            max_profit = trade_setup['max_profit_per_contract']
            profit_per_contract = np.random.uniform(max_profit * 0.3, max_profit * 0.8)
            return profit_per_contract * contracts
        else:
            # Loss trade - partial loss
            max_loss = trade_setup['max_loss_per_contract']
            loss_per_contract = -np.random.uniform(max_loss * 0.2, max_loss * 0.7)
            return loss_per_contract * contracts
    
    def get_vix_data(self, date: datetime) -> float:
        """Get realistic VIX data."""
        # Simulate realistic VIX range (10-30)
        base_vix = 16.0
        day_of_year = date.timetuple().tm_yday
        vix_value = base_vix + 3 * np.sin(day_of_year / 30) + np.random.normal(0, 2)
        return max(10.0, min(30.0, vix_value))
    
    def get_spy_price(self, date: datetime) -> float:
        """Get realistic SPY price."""
        days_from_start = (date - datetime(2024, 6, 13)).days
        base_price = 550.0
        trend = 0.1 * days_from_start
        volatility = 3 * np.sin(days_from_start / 7) + np.random.normal(0, 2)
        return base_price + trend + volatility
    
    def execute_strategy(self, date: datetime, vix: float, spy_price: float) -> dict:
        """Execute realistic strategy."""
        # Reset daily P&L on new day
        if self.current_date != date:
            self.daily_pnl = 0.0
            self.current_date = date
        
        # Check if we've hit daily limits
        if self.daily_pnl >= self.daily_profit_target:
            return None  # Target achieved, stop trading
        
        if self.daily_pnl <= -self.max_daily_loss:
            return None  # Daily loss limit hit, stop trading
        
        # Choose strategy based on VIX regime (your original logic)
        if vix > self.vix_high:
            # HIGH VIX: Sell premium
            if np.random.random() > 0.5:
                trade_setup = self.strategy.iron_condor(spy_price)
            else:
                trade_setup = self.strategy.iron_butterfly(spy_price)
        elif vix < self.vix_low:
            # LOW VIX: Buy premium
            trade_setup = self.strategy.diagonal_spread(spy_price)
        else:
            # NEUTRAL: No trade
            return None
        
        # Calculate realistic position size
        contracts = self.calculate_realistic_position_size(trade_setup)
        
        # Simulate realistic outcome
        total_pnl = self.simulate_realistic_outcome(trade_setup, contracts)
        
        # Update account and daily P&L
        self.current_account_value += total_pnl
        self.daily_pnl += total_pnl
        
        # Calculate metrics
        risk_amount = contracts * trade_setup['max_loss_per_contract']
        risk_percentage = (risk_amount / self.current_account_value) * 100
        
        return {
            'date': date.strftime('%Y-%m-%d'),
            'vix': vix,
            'spy_price': spy_price,
            'strategy': trade_setup['strategy'],
            'contracts': contracts,
            'pnl': total_pnl,
            'daily_pnl': self.daily_pnl,
            'account_value': self.current_account_value,
            'risk_amount': risk_amount,
            'risk_percentage': risk_percentage,
            'max_loss_per_contract': trade_setup['max_loss_per_contract'],
            'max_profit_per_contract': trade_setup['max_profit_per_contract']
        }
    
    def run_backtest(self, start_date: datetime, end_date: datetime):
        """Run realistic backtest."""
        print("🎯 REALISTIC 0DTE Options Strategy Backtest")
        print("=" * 60)
        print(f"Account Size: ${self.initial_account_value:,.2f}")
        print(f"Daily Target: ${self.daily_profit_target:.2f}")
        print(f"Max Risk Per Trade: ${self.max_risk_per_trade:.2f}")
        print(f"Max Daily Loss: ${self.max_daily_loss:.2f}")
        print("=" * 60)
        
        current_date = start_date
        trading_days = 0
        
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Trading days only
                vix_value = self.get_vix_data(current_date)
                spy_price = self.get_spy_price(current_date)
                
                trade_result = self.execute_strategy(current_date, vix_value, spy_price)
                
                if trade_result:
                    self.trades.append(trade_result)
                    trading_days += 1
                    
                    print(f"📅 {current_date.strftime('%Y-%m-%d')} | "
                          f"VIX: {vix_value:.2f} | "
                          f"SPY: ${spy_price:.2f} | "
                          f"Strategy: {trade_result['strategy']} | "
                          f"Contracts: {trade_result['contracts']} | "
                          f"P&L: ${trade_result['pnl']:,.2f} | "
                          f"Risk: {trade_result['risk_percentage']:.1f}%")
            
            current_date += timedelta(days=1)
        
        # Results
        trades_df = pd.DataFrame(self.trades)
        self.display_realistic_results(trades_df)
        
        # Save results
        trades_df.to_csv('realistic_zero_dte_trades.csv', index=False)
        print(f"\n📊 Realistic trade log saved as 'realistic_zero_dte_trades.csv'")
        
        return trades_df
    
    def display_realistic_results(self, trades_df: pd.DataFrame):
        """Display realistic results."""
        if trades_df.empty:
            print("No trades executed")
            return
        
        final_account = self.current_account_value
        total_return = (final_account - self.initial_account_value) / self.initial_account_value
        total_pnl = trades_df['pnl'].sum()
        
        print("\n" + "=" * 60)
        print("📊 REALISTIC BACKTEST RESULTS")
        print("=" * 60)
        
        print(f"💰 ACCOUNT PERFORMANCE:")
        print(f"   Initial Account: ${self.initial_account_value:,.2f}")
        print(f"   Final Account: ${final_account:,.2f}")
        print(f"   Total Return: {total_return:.2%}")
        print(f"   Total P&L: ${total_pnl:,.2f}")
        
        print(f"\n🎯 TRADING METRICS:")
        print(f"   Total Trades: {len(trades_df)}")
        print(f"   Win Rate: {(trades_df['pnl'] > 0).mean():.1%}")
        print(f"   Average P&L: ${trades_df['pnl'].mean():.2f}")
        print(f"   Best Day: ${trades_df['pnl'].max():.2f}")
        print(f"   Worst Day: ${trades_df['pnl'].min():.2f}")
        print(f"   Average Contracts: {trades_df['contracts'].mean():.1f}")
        print(f"   Max Contracts: {trades_df['contracts'].max()}")
        print(f"   Average Risk: {trades_df['risk_percentage'].mean():.1f}%")
        
        print(f"\n📈 DAILY PERFORMANCE:")
        daily_target_hits = len(trades_df[trades_df['pnl'] >= self.daily_profit_target])
        print(f"   Daily Target Hits: {daily_target_hits}/{len(trades_df)}")
        print(f"   Target Achievement: {(daily_target_hits/len(trades_df)):.1%}")
        
        print(f"\n🎯 STRATEGY BREAKDOWN:")
        for strategy in trades_df['strategy'].unique():
            strategy_data = trades_df[trades_df['strategy'] == strategy]
            print(f"   {strategy.upper()}:")
            print(f"     Trades: {len(strategy_data)}")
            print(f"     Total P&L: ${strategy_data['pnl'].sum():.2f}")
            print(f"     Win Rate: {(strategy_data['pnl'] > 0).mean():.1%}")
            print(f"     Avg Contracts: {strategy_data['contracts'].mean():.1f}")

def main():
    """Run realistic backtest."""
    backtest = RealisticBacktester()
    start_date = datetime(2024, 6, 13)
    end_date = datetime(2024, 7, 13)
    
    trades_df = backtest.run_backtest(start_date, end_date)
    return trades_df

if __name__ == "__main__":
    main()
