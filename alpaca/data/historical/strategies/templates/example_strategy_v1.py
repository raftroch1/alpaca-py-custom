#!/usr/bin/env python3
"""
EXAMPLE STRATEGY V1
Template example showing how to inherit from BaseThetaStrategy.

This is a simple contrarian strategy that:
- Goes short when VIX > 20 (sell straddles)
- Goes long when VIX < 15 (buy protective puts)
- Stays neutral when VIX 15-20

Usage:
    python example_strategy_v1.py
"""

import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Import the base template
sys.path.append(os.path.dirname(__file__))
from base_theta_strategy import BaseThetaStrategy


class ExampleStrategy(BaseThetaStrategy):
    """
    Example strategy implementing the BaseThetaStrategy template.
    
    This demonstrates proper inheritance and implementation of required methods.
    """
    
    def __init__(self):
        super().__init__(
            strategy_name="example_strategy",
            version="v1",
            starting_capital=25000,
            max_risk_per_trade=0.03,
            target_profit_per_trade=0.002
        )
        
        # Strategy-specific parameters
        self.high_vix_threshold = 20
        self.low_vix_threshold = 15
    
    def analyze_market_conditions(self, spy_price: float, vix_level: float, date: str) -> Dict[str, Any]:
        """Analyze market conditions and determine strategy"""
        
        if vix_level > self.high_vix_threshold:
            regime = "HIGH_VIX"
            strategy_type = "SHORT_STRADDLE"
        elif vix_level < self.low_vix_threshold:
            regime = "LOW_VIX"
            strategy_type = "PROTECTIVE_PUT"
        else:
            regime = "NEUTRAL_VIX"
            strategy_type = "NO_TRADE"
        
        return {
            'regime': regime,
            'strategy_type': strategy_type,
            'vix_level': vix_level,
            'spy_price': spy_price,
            'should_trade': strategy_type != "NO_TRADE"
        }
    
    def execute_strategy(self, market_analysis: Dict[str, Any], spy_price: float, date: str) -> Optional[Dict[str, Any]]:
        """Execute the strategy based on market analysis"""
        
        if not market_analysis['should_trade']:
            self.skip_trade(f"VIX neutral ({market_analysis['vix_level']:.2f})", date)
            return None
        
        strategy_type = market_analysis['strategy_type']
        date_formatted = datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d')
        
        if strategy_type == "SHORT_STRADDLE":
            return self._execute_short_straddle(spy_price, date, date_formatted)
        elif strategy_type == "PROTECTIVE_PUT":
            return self._execute_protective_put(spy_price, date, date_formatted)
        
        return None
    
    def _execute_short_straddle(self, spy_price: float, date: str, date_formatted: str) -> Optional[Dict[str, Any]]:
        """Execute short straddle strategy"""
        
        # Calculate strikes (ATM)
        call_strike = round(spy_price)
        put_strike = round(spy_price)
        
        # Get option prices (same-day expiration for 0DTE)
        call_price = self.get_option_price('SPY', date_formatted, call_strike, 'C', date_formatted)
        put_price = self.get_option_price('SPY', date_formatted, put_strike, 'P', date_formatted)
        
        if not call_price or not put_price:
            self.skip_trade("No real option data available", date)
            return None
        
        # Calculate premium and position size
        total_premium = call_price + put_price
        contracts = self.calculate_position_size("SHORT_STRADDLE", total_premium)
        
        if contracts == 0:
            self.skip_trade("Position size calculation resulted in 0 contracts", date)
            return None
        
        # Simulate P&L (for demo - assume options expire worthless 70% of the time)
        import random
        if random.random() < 0.3:  # 30% chance of loss
            profit = -total_premium * contracts * 100 * 1.5  # Loss bigger than premium
        else:
            profit = total_premium * contracts * 100 * 0.8  # Keep 80% of premium
        
        return {
            'date': date,
            'strategy': 'SHORT_STRADDLE',
            'spy_price': spy_price,
            'call_strike': call_strike,
            'put_strike': put_strike,
            'call_price': call_price,
            'put_price': put_price,
            'contracts': contracts,
            'premium_collected': total_premium,
            'profit': profit
        }
    
    def _execute_protective_put(self, spy_price: float, date: str, date_formatted: str) -> Optional[Dict[str, Any]]:
        """Execute protective put strategy"""
        
        # Calculate strike (slightly OTM)
        put_strike = round(spy_price * 0.98)  # 2% OTM
        
        # Get option price
        put_price = self.get_option_price('SPY', date_formatted, put_strike, 'P', date_formatted)
        
        if not put_price:
            self.skip_trade("No real put option data available", date)
            return None
        
        # Calculate position size
        contracts = self.calculate_position_size("PROTECTIVE_PUT", put_price)
        
        if contracts == 0:
            self.skip_trade("Position size calculation resulted in 0 contracts", date)
            return None
        
        # Simulate P&L (for demo - protective puts usually expire worthless)
        import random
        if random.random() < 0.1:  # 10% chance of profit (market crash)
            profit = put_price * contracts * 100 * 5  # Big profit on crash
        else:
            profit = -put_price * contracts * 100  # Lose premium paid
        
        return {
            'date': date,
            'strategy': 'PROTECTIVE_PUT',
            'spy_price': spy_price,
            'put_strike': put_strike,
            'put_price': put_price,
            'contracts': contracts,
            'premium_paid': put_price,
            'profit': profit
        }
    
    def calculate_position_size(self, strategy_type: str, premium: float) -> int:
        """Calculate position size based on risk management"""
        
        if strategy_type == "SHORT_STRADDLE":
            # Risk = potential loss if option goes ITM
            max_risk_per_contract = premium * 100 * 2  # Assume max 2x premium loss
        elif strategy_type == "PROTECTIVE_PUT":
            # Risk = premium paid
            max_risk_per_contract = premium * 100
        else:
            return 0
        
        # Calculate max contracts based on risk per trade
        max_risk_total = self.current_capital * self.max_risk_per_trade
        max_contracts = int(max_risk_total / max_risk_per_contract)
        
        # Ensure at least 1 contract if we can afford it
        return max(1, max_contracts) if max_risk_per_contract <= max_risk_total else 0


def main():
    """Run the example strategy"""
    print("🚀 Running Example Strategy V1...")
    
    strategy = ExampleStrategy()
    
    # Run backtest for recent period
    strategy.run_backtest('2025-01-01', '2025-01-31')  # January 2025


if __name__ == "__main__":
    main() 