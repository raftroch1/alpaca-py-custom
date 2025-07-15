#!/usr/bin/env python3
"""
OPTIMIZED 0DTE Options Strategy Backtest

Key Fixes from Enhanced Version:
1. Fixed VIX percentile calculation with better historical data handling
2. More aggressive but smart strategy selection for more trading opportunities
3. Improved regime detection using actual VIX values instead of percentiles
4. Better balance between risk and reward
5. Dynamic adjustment based on recent performance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedMarketAnalyzer:
    """Optimized market analysis with improved calculations."""
    
    def __init__(self):
        self.vix_history = []
        self.spy_history = []
        self.dates = []
        
        # Use historical VIX statistics for better regime detection
        self.vix_stats = {
            'low_threshold': 15.0,    # Historically low VIX
            'high_threshold': 20.0,   # Historically high VIX  
            'extreme_low': 12.0,      # Extremely low VIX
            'extreme_high': 25.0      # Extremely high VIX
        }
    
    def add_market_data(self, date: datetime, vix: float, spy_price: float):
        """Add market data for analysis."""
        self.dates.append(date)
        self.vix_history.append(vix)
        self.spy_history.append(spy_price)
    
    def get_vix_regime(self, current_vix: float) -> str:
        """Get VIX regime based on historical levels."""
        if current_vix <= self.vix_stats['extreme_low']:
            return 'extreme_low'
        elif current_vix <= self.vix_stats['low_threshold']:
            return 'low'
        elif current_vix >= self.vix_stats['extreme_high']:
            return 'extreme_high'
        elif current_vix >= self.vix_stats['high_threshold']:
            return 'high'
        else:
            return 'neutral'
    
    def calculate_vix_trend(self, lookback: int = 5) -> str:
        """Calculate VIX trend over recent periods."""
        if len(self.vix_history) < lookback:
            return 'neutral'
        
        recent_vix = self.vix_history[-lookback:]
        if len(recent_vix) < 2:
            return 'neutral'
        
        trend = recent_vix[-1] - recent_vix[0]
        if trend > 1:
            return 'rising'
        elif trend < -1:
            return 'falling'
        else:
            return 'neutral'
    
    def calculate_spy_momentum(self, period: int = 3) -> float:
        """Calculate SPY momentum over period."""
        if len(self.spy_history) < period + 1:
            return 0.0
        
        return (self.spy_history[-1] - self.spy_history[-period-1]) / self.spy_history[-period-1] * 100
    
    def get_market_stress_level(self, current_vix: float) -> float:
        """Get market stress level (0-1 scale)."""
        # Normalize VIX to stress level
        if current_vix <= 12:
            return 0.1  # Very low stress
        elif current_vix <= 16:
            return 0.3  # Low stress
        elif current_vix <= 20:
            return 0.5  # Moderate stress
        elif current_vix <= 25:
            return 0.7  # High stress
        else:
            return 0.9  # Very high stress

class OptimizedOptionsStrategy:
    """Optimized options strategy with improved market-based outcomes."""
    
    def __init__(self):
        self.commission = 0.65
        
        # Strategy performance tracking
        self.strategy_performance = {
            'iron_condor': {'wins': 0, 'losses': 0, 'total_pnl': 0.0},
            'iron_butterfly': {'wins': 0, 'losses': 0, 'total_pnl': 0.0},
            'diagonal_spread': {'wins': 0, 'losses': 0, 'total_pnl': 0.0}
        }
    
    def update_strategy_performance(self, strategy: str, pnl: float):
        """Update strategy performance tracking."""
        if pnl > 0:
            self.strategy_performance[strategy]['wins'] += 1
        else:
            self.strategy_performance[strategy]['losses'] += 1
        self.strategy_performance[strategy]['total_pnl'] += float(pnl)
    
    def get_strategy_success_rate(self, strategy: str) -> float:
        """Get historical success rate for strategy."""
        perf = self.strategy_performance[strategy]
        total_trades = perf['wins'] + perf['losses']
        if total_trades == 0:
            return 0.65  # Default assumption
        return perf['wins'] / total_trades
    
    def get_optimized_outcome(self, strategy_type: str, market_conditions: Dict) -> Dict:
        """Get optimized market-based outcome probabilities."""
        
        vix_regime = market_conditions['vix_regime']
        vix_trend = market_conditions['vix_trend']
        momentum = market_conditions['spy_momentum']
        stress_level = market_conditions['stress_level']
        
        # Base probabilities adjusted for market conditions
        if strategy_type == 'iron_condor':
            base_prob = 0.68
            
            # Iron Condor performs best in low volatility, range-bound markets
            if vix_regime in ['low', 'extreme_low']:
                prob_adjustment = 0.12
            elif vix_regime == 'neutral':
                prob_adjustment = 0.05
            else:
                prob_adjustment = -0.10
            
            # Trending markets hurt credit spreads
            if abs(momentum) > 1.5:
                prob_adjustment -= 0.08
            
            # Falling VIX helps credit strategies
            if vix_trend == 'falling':
                prob_adjustment += 0.05
            elif vix_trend == 'rising':
                prob_adjustment -= 0.05
            
            credit_per_contract = np.random.uniform(1.80, 3.50)
            
        elif strategy_type == 'iron_butterfly':
            base_prob = 0.62
            
            # Iron Butterfly benefits from very low movement
            if vix_regime in ['low', 'extreme_low'] and abs(momentum) < 1.0:
                prob_adjustment = 0.15
            elif vix_regime == 'neutral':
                prob_adjustment = 0.08
            else:
                prob_adjustment = -0.12
            
            # Very sensitive to momentum
            if abs(momentum) > 2.0:
                prob_adjustment -= 0.15
            
            credit_per_contract = np.random.uniform(2.20, 4.50)
            
        elif strategy_type == 'diagonal_spread':
            base_prob = 0.58
            
            # Diagonal spreads work well with moderate volatility and directional movement
            if vix_regime in ['neutral', 'low'] and abs(momentum) > 0.5:
                prob_adjustment = 0.12
            elif vix_regime == 'high' and abs(momentum) > 1.0:
                prob_adjustment = 0.08
            else:
                prob_adjustment = -0.05
            
            # Benefits from trending markets
            if abs(momentum) > 1.5:
                prob_adjustment += 0.08
            
            credit_per_contract = np.random.uniform(1.20, 3.20)
        
        else:
            base_prob = 0.50
            prob_adjustment = 0
            credit_per_contract = 1.50
        
        # Apply historical performance adjustment
        historical_rate = self.get_strategy_success_rate(strategy_type)
        if historical_rate > 0:
            # Blend historical performance (30%) with market-based prediction (70%)
            final_prob = 0.7 * (base_prob + prob_adjustment) + 0.3 * historical_rate
        else:
            final_prob = base_prob + prob_adjustment
        
        # Ensure reasonable bounds
        final_prob = max(0.35, min(0.80, final_prob))
        
        return {
            'profit_prob': final_prob,
            'credit_per_contract': credit_per_contract,
            'max_loss_per_contract': 5.0 - credit_per_contract,
            'max_profit_per_contract': credit_per_contract - (4 * self.commission)
        }
    
    def iron_condor(self, market_conditions: Dict) -> Dict:
        """Optimized Iron Condor."""
        outcome = self.get_optimized_outcome('iron_condor', market_conditions)
        return {'strategy': 'iron_condor', **outcome}
    
    def iron_butterfly(self, market_conditions: Dict) -> Dict:
        """Optimized Iron Butterfly."""
        outcome = self.get_optimized_outcome('iron_butterfly', market_conditions)
        return {'strategy': 'iron_butterfly', **outcome}
    
    def diagonal_spread(self, market_conditions: Dict) -> Dict:
        """Optimized Diagonal Spread."""
        outcome = self.get_optimized_outcome('diagonal_spread', market_conditions)
        return {'strategy': 'diagonal_spread', **outcome}

class OptimizedBacktester:
    """Optimized backtester with improved logic."""
    
    def __init__(self):
        self.strategy = OptimizedOptionsStrategy()
        self.analyzer = OptimizedMarketAnalyzer()
        self.trades = []
        
        # Account parameters  
        self.initial_account_value = 25000.0
        self.current_account_value = self.initial_account_value
        self.daily_profit_target = 250.0
        self.max_risk_per_trade = 500.0
        self.max_daily_loss = 750.0
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.current_date = None
        self.trade_count_today = 0
        self.max_trades_per_day = 2
        
        # Performance tracking
        self.recent_performance = []  # Track last 10 days
        self.consecutive_losses = 0
        
    def calculate_adaptive_position_size(self, trade_setup: Dict, market_conditions: Dict) -> int:
        """Calculate position size that adapts to recent performance."""
        base_risk = self.max_risk_per_trade
        
        # Reduce risk after consecutive losses
        if self.consecutive_losses >= 3:
            base_risk *= 0.5
        elif self.consecutive_losses >= 2:
            base_risk *= 0.75
        
        # Reduce risk in high stress environments
        stress_level = market_conditions['stress_level']
        if stress_level > 0.7:
            base_risk *= 0.6
        elif stress_level > 0.5:
            base_risk *= 0.8
        
        # Calculate contracts
        max_loss_per_contract = trade_setup['max_loss_per_contract']
        if max_loss_per_contract <= 0:
            return 1
            
        contracts = int(base_risk / max_loss_per_contract)
        
        # Apply bounds
        min_contracts = 1
        max_contracts = min(50, contracts)
        
        # Ensure reasonable position size for account
        if self.current_account_value < 30000:
            max_contracts = min(30, max_contracts)
        
        return max(min_contracts, max_contracts)
    
    def simulate_optimized_outcome(self, trade_setup: Dict, contracts: int, market_conditions: Dict) -> float:
        """Simulate optimized market-based outcome."""
        profit_prob = trade_setup['profit_prob']
        
        # Determine outcome
        is_winner = np.random.random() < profit_prob
        
        if is_winner:
            # Winning trade - vary profit capture
            max_profit = trade_setup['max_profit_per_contract']
            
            # In low stress, high prob environments, capture more profit
            stress_level = market_conditions['stress_level']
            if stress_level < 0.3 and profit_prob > 0.7:
                profit_capture = np.random.uniform(0.6, 0.9)
            else:
                profit_capture = np.random.uniform(0.4, 0.7)
            
            profit_per_contract = max_profit * profit_capture
            total_pnl = profit_per_contract * contracts
            
        else:
            # Losing trade - vary loss amount
            max_loss = trade_setup['max_loss_per_contract']
            
            # In high stress environments, losses tend to be larger
            stress_level = market_conditions['stress_level']
            if stress_level > 0.7:
                loss_percentage = np.random.uniform(0.6, 0.9)
            else:
                loss_percentage = np.random.uniform(0.3, 0.6)
            
            loss_per_contract = -max_loss * loss_percentage
            total_pnl = loss_per_contract * contracts
        
        # Update strategy performance
        self.strategy.update_strategy_performance(trade_setup['strategy'], total_pnl)
        
        return total_pnl
    
    def get_market_conditions(self, date: datetime, vix: float, spy_price: float) -> Dict:
        """Get comprehensive market conditions."""
        self.analyzer.add_market_data(date, vix, spy_price)
        
        return {
            'vix': vix,
            'vix_regime': self.analyzer.get_vix_regime(vix),
            'vix_trend': self.analyzer.calculate_vix_trend(),
            'spy_price': spy_price,
            'spy_momentum': self.analyzer.calculate_spy_momentum(),
            'stress_level': self.analyzer.get_market_stress_level(vix)
        }
    
    def select_optimal_strategy(self, market_conditions: Dict) -> Optional[Dict]:
        """Select optimal strategy based on market conditions."""
        vix = market_conditions['vix']
        vix_regime = market_conditions['vix_regime']
        vix_trend = market_conditions['vix_trend']
        momentum = market_conditions['spy_momentum']
        stress_level = market_conditions['stress_level']
        
        # Filter extreme conditions
        if vix > 35 or vix < 10:
            return None  # Too extreme
        
        # Strategy selection logic
        if vix_regime in ['extreme_low', 'low']:
            # LOW VOLATILITY environment
            if abs(momentum) < 1.0 and vix_trend != 'rising':
                # Range-bound, stable - perfect for credit spreads
                if vix < 13:
                    return self.strategy.iron_butterfly(market_conditions)  # Very tight range
                else:
                    return self.strategy.iron_condor(market_conditions)      # Moderate range
                    
            elif abs(momentum) > 0.8:
                # Some directional movement - good for diagonal spreads
                return self.strategy.diagonal_spread(market_conditions)
        
        elif vix_regime == 'neutral':
            # NEUTRAL VOLATILITY environment  
            if abs(momentum) < 1.5 and stress_level < 0.6:
                # Moderate conditions - can trade credit spreads
                return self.strategy.iron_condor(market_conditions)
                
            elif abs(momentum) > 1.0:
                # Trending market - diagonal spreads
                return self.strategy.diagonal_spread(market_conditions)
        
        elif vix_regime == 'high':
            # HIGH VOLATILITY environment
            if vix_trend == 'falling' and abs(momentum) < 2.0:
                # Volatility coming down, range-bound - sell premium
                return self.strategy.iron_condor(market_conditions)
                
            elif abs(momentum) > 1.5:
                # High vol with trend - be careful, maybe diagonal
                if stress_level < 0.7:  # Not too stressed
                    return self.strategy.diagonal_spread(market_conditions)
        
        elif vix_regime == 'extreme_high':
            # EXTREME VOLATILITY - be very selective
            if vix_trend == 'falling' and abs(momentum) < 1.0:
                # Vol spike cooling down - opportunity to sell premium
                return self.strategy.iron_condor(market_conditions)
        
        # No suitable conditions
        return None
    
    def execute_strategy(self, date: datetime, vix: float, spy_price: float) -> Optional[Dict]:
        """Execute optimized strategy."""
        # Reset daily tracking
        if self.current_date != date:
            if self.daily_pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
                
            self.daily_pnl = 0.0
            self.trade_count_today = 0
            self.current_date = date
        
        # Check daily limits
        if (self.daily_pnl >= self.daily_profit_target or 
            self.daily_pnl <= -self.max_daily_loss or
            self.trade_count_today >= self.max_trades_per_day):
            return None
        
        # Get market conditions
        market_conditions = self.get_market_conditions(date, vix, spy_price)
        
        # Select strategy
        trade_setup = self.select_optimal_strategy(market_conditions)
        if not trade_setup:
            return None
        
        # Calculate position size
        contracts = self.calculate_adaptive_position_size(trade_setup, market_conditions)
        
        # Simulate outcome
        total_pnl = self.simulate_optimized_outcome(trade_setup, contracts, market_conditions)
        
        # Update tracking
        self.current_account_value += total_pnl
        self.daily_pnl += total_pnl
        self.trade_count_today += 1
        
        # Calculate metrics
        risk_amount = contracts * trade_setup['max_loss_per_contract']
        risk_percentage = (risk_amount / self.current_account_value) * 100
        
        return {
            'date': date.strftime('%Y-%m-%d'),
            'vix': vix,
            'vix_regime': market_conditions['vix_regime'],
            'vix_trend': market_conditions['vix_trend'],
            'spy_price': spy_price,
            'spy_momentum': market_conditions['spy_momentum'],
            'stress_level': market_conditions['stress_level'],
            'strategy': trade_setup['strategy'],
            'contracts': contracts,
            'pnl': total_pnl,
            'daily_pnl': self.daily_pnl,
            'account_value': self.current_account_value,
            'risk_amount': risk_amount,
            'risk_percentage': risk_percentage,
            'profit_prob': trade_setup['profit_prob'],
            'consecutive_losses': self.consecutive_losses
        }
    
    def run_backtest(self, start_date: datetime, end_date: datetime):
        """Run optimized backtest."""
        print("🔥 OPTIMIZED 0DTE Options Strategy Backtest")
        print("=" * 60)
        print(f"Account Size: ${self.initial_account_value:,.2f}")
        print(f"Daily Target: ${self.daily_profit_target:,.2f}")
        print(f"Max Risk Per Trade: ${self.max_risk_per_trade:,.2f}")
        print(f"Max Daily Loss: ${self.max_daily_loss:,.2f}")
        print("=" * 60)
        
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Skip weekends
                vix = self.get_vix_data(current_date)
                spy_price = self.get_spy_price(current_date)
                
                trade_result = self.execute_strategy(current_date, vix, spy_price)
                
                if trade_result:
                    self.trades.append(trade_result)
                    print(f"📅 {trade_result['date']} | VIX: {trade_result['vix']:.2f} ({trade_result['vix_regime']}) | "
                          f"Trend: {trade_result['vix_trend']} | Momentum: {trade_result['spy_momentum']:.1f}% | "
                          f"Strategy: {trade_result['strategy']} | Contracts: {trade_result['contracts']} | "
                          f"P&L: ${trade_result['pnl']:.2f} | Risk: {trade_result['risk_percentage']:.1f}%")
            
            current_date += timedelta(days=1)
        
        if self.trades:
            self.display_optimized_results()
        else:
            print("❌ No trades executed during the backtest period.")
    
    def get_vix_data(self, date: datetime) -> float:
        """Get realistic VIX data with better patterns."""
        base_vix = 16.0
        day_of_year = date.timetuple().tm_yday
        
        # Market cycles and events
        if 50 <= day_of_year <= 70:  # Late Feb/Early March - often volatile
            base_vix += 3
        elif 120 <= day_of_year <= 140:  # Late April/May - earnings season
            base_vix += 2
        elif 200 <= day_of_year <= 220:  # Late July/August - summer lull
            base_vix -= 2
        elif 280 <= day_of_year <= 300:  # October - historically volatile
            base_vix += 4
        
        # Add realistic randomness
        daily_change = np.random.normal(0, 1.2)
        vix_value = base_vix + daily_change
        
        return max(10.0, min(35.0, vix_value))
    
    def get_spy_price(self, date: datetime) -> float:
        """Get realistic SPY price with trends."""
        base_price = 450.0
        days_from_start = (date - datetime(2024, 1, 1)).days
        
        # Bull market trend
        trend = days_from_start * 0.08
        
        # Market cycles
        cyclical = 15 * np.sin(days_from_start / 60)
        
        # Realistic daily volatility
        daily_change = np.random.normal(0, 3.5)
        
        price = base_price + trend + cyclical + daily_change
        return max(350.0, min(650.0, price))
    
    def display_optimized_results(self):
        """Display comprehensive optimized results."""
        trades_df = pd.DataFrame(self.trades)
        
        total_return = (self.current_account_value - self.initial_account_value) / self.initial_account_value * 100
        total_pnl = self.current_account_value - self.initial_account_value
        
        print("\n" + "=" * 60)
        print("🔥 OPTIMIZED BACKTEST RESULTS")
        print("=" * 60)
        print(f"💰 ACCOUNT PERFORMANCE:")
        print(f"   Initial Account: ${self.initial_account_value:,.2f}")
        print(f"   Final Account: ${self.current_account_value:,.2f}")
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Total P&L: ${total_pnl:,.2f}")
        
        print(f"\n🎯 TRADING METRICS:")
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        print(f"   Total Trades: {total_trades}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Average P&L: ${trades_df['pnl'].mean():.2f}")
        print(f"   Best Trade: ${trades_df['pnl'].max():.2f}")
        print(f"   Worst Trade: ${trades_df['pnl'].min():.2f}")
        print(f"   Average Contracts: {trades_df['contracts'].mean():.1f}")
        print(f"   Average Risk: {trades_df['risk_percentage'].mean():.1f}%")
        
        # VIX regime analysis
        regime_stats = trades_df.groupby('vix_regime').agg({
            'pnl': ['count', 'sum', 'mean']
        }).round(2)
        
        print(f"\n📊 VIX REGIME PERFORMANCE:")
        for regime in regime_stats.index:
            count = regime_stats.loc[regime, ('pnl', 'count')]
            total_pnl_regime = regime_stats.loc[regime, ('pnl', 'sum')]
            avg_pnl = regime_stats.loc[regime, ('pnl', 'mean')]
            print(f"   {regime.upper()}: {count} trades, ${total_pnl_regime:.2f} total, ${avg_pnl:.2f} avg")
        
        # Strategy performance
        strategy_stats = trades_df.groupby('strategy').agg({
            'pnl': ['count', 'sum', 'mean'],
            'contracts': 'mean'
        }).round(2)
        
        print(f"\n🎯 STRATEGY BREAKDOWN:")
        for strategy in strategy_stats.index:
            count = strategy_stats.loc[strategy, ('pnl', 'count')]
            total_pnl_strat = strategy_stats.loc[strategy, ('pnl', 'sum')]
            avg_pnl = strategy_stats.loc[strategy, ('pnl', 'mean')]
            avg_contracts = strategy_stats.loc[strategy, ('contracts', 'mean')]
            strategy_wins = len(trades_df[(trades_df['strategy'] == strategy) & (trades_df['pnl'] > 0)])
            strategy_win_rate = (strategy_wins / count * 100) if count > 0 else 0
            
            print(f"   {strategy.upper()}:")
            print(f"     Trades: {count}")
            print(f"     Total P&L: ${total_pnl_strat:.2f}")
            print(f"     Avg P&L: ${avg_pnl:.2f}")
            print(f"     Win Rate: {strategy_win_rate:.1f}%")
            print(f"     Avg Contracts: {avg_contracts:.1f}")
        
        # Save results
        trades_df.to_csv('optimized_zero_dte_trades.csv', index=False)
        print(f"\n📊 Optimized trade log saved as 'optimized_zero_dte_trades.csv'")

def main():
    """Run optimized backtest."""
    backtester = OptimizedBacktester()
    
    start_date = datetime(2024, 6, 13)
    end_date = datetime(2024, 7, 13)
    
    backtester.run_backtest(start_date, end_date)

if __name__ == "__main__":
    main() 