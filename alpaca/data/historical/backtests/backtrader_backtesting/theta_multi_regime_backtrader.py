#!/usr/bin/env python3
"""
THETADATA MULTI-REGIME OPTIONS STRATEGY BACKTRADER
A comprehensive backtest integrating Backtrader with proven ThetaData infrastructure

Features:
- Real option pricing from ThetaData API (proven working format)
- Multi-regime strategy logic (Iron Condor, Diagonal, Credit Spreads, Iron Butterfly)
- Proper position sizing with 100-share multiplier
- Kelly criterion-based position sizing
- Comprehensive risk management
- Professional logging and performance tracking

Strategy Regimes:
- High VIX + Rising: Iron Condor
- Low VIX: Diagonal Spread
- Moderate VIX + Bullish: Put Credit Spread
- Moderate VIX + Bearish: Call Credit Spread
- Moderate VIX + Neutral: Iron Butterfly
"""

import sys
import os
import pandas as pd
import numpy as np
import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
import yfinance as yf
from dotenv import load_dotenv
import backtrader as bt
import math

# Add alpaca imports for SPY data
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

warnings.filterwarnings('ignore')

# Import the shared connector
from alpaca.data.historical.thetadata.connector import ThetaDataConnector

class RiskManager:
    """
    Enhanced risk management with Kelly criterion and strategy-specific adjustments
    """
    
    def __init__(self, max_risk_per_trade=0.02, max_daily_loss=0.03, max_trades_per_day=3, kelly_fraction=0.25):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_trades_per_day = max_trades_per_day
        self.kelly_fraction = kelly_fraction
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.daily_start_value = None
        
        # Historical performance data for Kelly calculation
        self.win_rate = 0.65
        self.win_loss_ratio = 1.5
        
        # Strategy-specific multipliers
        self.strategy_multipliers = {
            "IRON_CONDOR": 1.0,
            "DIAGONAL": 0.6,
            "PUT_CREDIT_SPREAD": 0.8,
            "CALL_CREDIT_SPREAD": 0.8,
            "IRON_BUTTERFLY": 0.7
        }
    
    def calculate_kelly_position(self, portfolio_value: float, strategy_type: str, max_risk: float) -> int:
        """Calculate position size using Kelly criterion"""
        # Kelly formula: f* = (bp - q) / b
        b = self.win_loss_ratio
        p = self.win_rate
        q = 1 - p
        kelly = (b * p - q) / b
        kelly = max(kelly, 0.01)  # Never go below 1%
        kelly *= self.kelly_fraction  # Use fractional Kelly for safety
        
        # Apply strategy-specific multiplier
        multiplier = self.strategy_multipliers.get(strategy_type, 1.0)
        
        # Calculate position size
        size = int((portfolio_value * kelly * multiplier) // max_risk)
        return max(size, 1)
    
    def can_trade(self, portfolio_value: float) -> bool:
        """Check if we can place a new trade based on risk limits"""
        if self.daily_pnl < -self.max_daily_loss * portfolio_value:
            return False
        if self.trades_today >= self.max_trades_per_day:
            return False
        return True
    
    def reset_daily_counters(self):
        """Reset daily counters for new trading day"""
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.daily_start_value = None

def safe_sum(*args):
    return sum(float(x) if x is not None else 0.0 for x in args)

class MultiRegimeOptionsStrategy(bt.Strategy):
    """
    Multi-regime options strategy for Backtrader using real ThetaData
    """
    
    params = (
        ('low_vol_threshold', 17),
        ('high_vol_threshold', 18),
        ('starting_capital', 25000),
        ('log_trades', True),
    )
    
    def __init__(self):
        # Data feeds
        self.spy = self.datas[0]  # SPY price data
        self.vix = self.datas[1]  # VIX data
        
        # Initialize components
        self.theta_connector = ThetaDataConnector()
        self.risk_manager = RiskManager()
        
        # Strategy state
        self.current_positions = {}
        self.trade_log = []
        self.daily_pnl = 0.0
        self.last_trade_date = None
        
        # Test ThetaData connection
        if not self.theta_connector.test_connection():
            print("⚠️  ThetaData connection failed - using simulation mode")
            self.use_theta_data = False
        else:
            print("✅ ThetaData connection successful")
            self.use_theta_data = True
        
        # Initialize logging
        if self.p.log_trades:
            self.setup_logging()
    
    def setup_logging(self):
        """Setup trade logging"""
        log_dir = os.path.join(os.path.dirname(__file__), 'logs', datetime.now().strftime('%Y-%m-%d'))
        os.makedirs(log_dir, exist_ok=True)
        
        self.trade_log_file = os.path.join(log_dir, f"theta_backtrader_trades_{datetime.now().strftime('%H%M%S')}.csv")
        
        # Create log file with headers
        with open(self.trade_log_file, 'w') as f:
            f.write("date,strategy,spy_price,vix,strike,right,option_price,contracts,premium,pnl,portfolio_value\n")
    
    def log_trade(self, trade_data: Dict):
        """Log trade to CSV file"""
        def safe_fmt(val, fmt=".2f"):
            try:
                return format(float(val), fmt)
            except Exception:
                return ""
        if self.p.log_trades:
            with open(self.trade_log_file, 'a') as f:
                f.write(f"{trade_data.get('date','')},{trade_data.get('strategy','')},{safe_fmt(trade_data.get('spy_price'))},"
                       f"{safe_fmt(trade_data.get('vix'))},{trade_data.get('strike','')},{trade_data.get('right','')},"
                       f"{safe_fmt(trade_data.get('option_price'))},{trade_data.get('contracts','')},{safe_fmt(trade_data.get('premium'))},"
                       f"{safe_fmt(trade_data.get('pnl'))},{safe_fmt(trade_data.get('portfolio_value'))}\n")
    
    def analyze_market_conditions(self) -> Tuple[str, Dict]:
        """
        Analyze market conditions to determine strategy regime
        Returns: (strategy_name, conditions_dict)
        """
        current_vix = self.vix.close[0]
        previous_vix = self.vix.close[-1] if len(self.vix.close) > 1 else current_vix
        spy_price = self.spy.close[0]
        
        # Calculate momentum (simplified)
        if len(self.spy.close) >= 5:
            recent_returns = [(self.spy.close[-i] / self.spy.close[-i-1] - 1) for i in range(1, 5)]
            momentum = np.mean(recent_returns)
        else:
            momentum = 0
        
        conditions = {
            'current_vix': current_vix,
            'previous_vix': previous_vix,
            'vix_higher': current_vix > previous_vix,
            'low_volatility': current_vix < self.p.low_vol_threshold,
            'moderate_volatility': self.p.low_vol_threshold <= current_vix <= self.p.high_vol_threshold,
            'high_volatility': current_vix > self.p.high_vol_threshold,
            'bullish_momentum': momentum > 0.01,  # 1% threshold
            'bearish_momentum': momentum < -0.01,
            'spy_price': spy_price
        }
        
        # Strategy selection logic (from reenhanced strategy)
        if conditions['vix_higher'] and conditions['high_volatility']:
            return "IRON_CONDOR", conditions
        elif conditions['low_volatility']:
            return "DIAGONAL", conditions
        elif conditions['moderate_volatility']:
            if conditions['bullish_momentum']:
                return "PUT_CREDIT_SPREAD", conditions
            elif conditions['bearish_momentum']:
                return "CALL_CREDIT_SPREAD", conditions
            else:
                return "IRON_BUTTERFLY", conditions
        else:
            return "NO_TRADE", conditions
    
    def calculate_position_size(self, strategy_type: str, option_price: float, portfolio_value: float) -> int:
        """Calculate position size with proper risk management"""
        # Account for 100-share multiplier
        cost_per_contract = option_price * 100
        
        # Maximum risk per trade
        max_risk_per_trade = portfolio_value * self.risk_manager.max_risk_per_trade
        
        # Calculate max contracts by risk
        max_contracts_by_risk = int(max_risk_per_trade / cost_per_contract) if cost_per_contract > 0 else 1
        
        # Use Kelly criterion for position sizing
        kelly_size = self.risk_manager.calculate_kelly_position(portfolio_value, strategy_type, cost_per_contract)
        
        # Final position size (conservative approach)
        position_size = min(max_contracts_by_risk, kelly_size, 10)  # Cap at 10 contracts
        
        return max(1, position_size)
    
    def execute_strategy(self, strategy: str, conditions: Dict) -> Optional[Dict]:
        """
        Execute the selected strategy with real option pricing
        """
        if strategy == "NO_TRADE":
            return None
        
        date_str = self.datetime.date().strftime('%Y-%m-%d')
        spy_price = conditions['spy_price']
        vix = conditions['current_vix']
        portfolio_value = self.broker.get_value()
        
        # Check risk limits
        if not self.risk_manager.can_trade(portfolio_value):
            return None
        
        trade_result = None
        
        if strategy == "IRON_CONDOR":
            trade_result = self.execute_iron_condor(date_str, spy_price, vix, portfolio_value)
        elif strategy == "PUT_CREDIT_SPREAD":
            trade_result = self.execute_put_credit_spread(date_str, spy_price, vix, portfolio_value)
        elif strategy == "CALL_CREDIT_SPREAD":
            trade_result = self.execute_call_credit_spread(date_str, spy_price, vix, portfolio_value)
        elif strategy == "IRON_BUTTERFLY":
            trade_result = self.execute_iron_butterfly(date_str, spy_price, vix, portfolio_value)
        elif strategy == "DIAGONAL":
            trade_result = self.execute_diagonal_spread(date_str, spy_price, vix, portfolio_value)
        
        if trade_result:
            # Update risk manager
            self.risk_manager.trades_today += 1
            self.risk_manager.daily_pnl += trade_result.get('pnl', 0)
            
            # Log trade
            self.log_trade(trade_result)
            self.trade_log.append(trade_result)
        
        return trade_result
    
    def execute_iron_condor(self, date: str, spy_price: float, vix: float, portfolio_value: float) -> Optional[Dict]:
        """Execute Iron Condor strategy using real option data"""
        if not self.use_theta_data:
            return self.simulate_iron_condor(date, spy_price, vix, portfolio_value)
        
        # Select strikes
        short_put_strike = round(spy_price * 0.95)  # 5% OTM
        long_put_strike = short_put_strike - 5
        short_call_strike = round(spy_price * 1.05)  # 5% OTM
        long_call_strike = short_call_strike + 5
        
        # Get real option prices
        short_put_price = self.theta_connector.get_option_price("SPY", date, short_put_strike, "P")
        long_put_price = self.theta_connector.get_option_price("SPY", date, long_put_strike, "P")
        short_call_price = self.theta_connector.get_option_price("SPY", date, short_call_strike, "C")
        long_call_price = self.theta_connector.get_option_price("SPY", date, long_call_strike, "C")
        
        # Check if all prices are available
        if any(x is None for x in [short_put_price, long_put_price, short_call_price, long_call_price]):
            return None
        
        # Calculate net credit
        net_credit = safe_sum(short_put_price, short_call_price) - safe_sum(long_put_price, long_call_price)
        
        if net_credit <= 0:
            return None  # Skip if not profitable
        
        try:
            avg_option_price = (float(short_put_price) + float(short_call_price)) / 2
        except Exception:
            return None
        
        contracts = self.calculate_position_size("IRON_CONDOR", avg_option_price, portfolio_value)
        
        # Calculate P&L (assuming expired worthless for credit strategies)
        premium_collected = net_credit * contracts * 100
        
        # Simulate the trade outcome (80% success rate for iron condors)
        is_profitable = np.random.random() < 0.80
        if is_profitable:
            pnl = premium_collected
        else:
            # Losing trade - assume assignment
            max_loss = (5 - net_credit) * contracts * 100  # Strike width - credit
            pnl = -max_loss
        
        # Execute the "trade" in backtrader (simplified as buy/sell SPY)
        if pnl > 0:
            self.buy(size=contracts)  # Represent profit as long position
        else:
            self.sell(size=abs(contracts))  # Represent loss as short position
        
        return {
            'date': date,
            'strategy': 'IRON_CONDOR',
            'spy_price': spy_price,
            'vix': vix,
            'strike': f"{short_put_strike}/{long_put_strike}/{short_call_strike}/{long_call_strike}",
            'right': 'IC',
            'option_price': net_credit,
            'contracts': contracts,
            'premium': premium_collected,
            'pnl': pnl,
            'portfolio_value': portfolio_value
        }
    
    def execute_put_credit_spread(self, date: str, spy_price: float, vix: float, portfolio_value: float) -> Optional[Dict]:
        """Execute Put Credit Spread using real option data"""
        if not self.use_theta_data:
            return self.simulate_put_credit_spread(date, spy_price, vix, portfolio_value)
        
        # Select strikes
        short_put_strike = round(spy_price * 0.95)  # 5% OTM
        long_put_strike = short_put_strike - 5
        
        # Get real option prices
        short_put_price = self.theta_connector.get_option_price("SPY", date, short_put_strike, "P")
        long_put_price = self.theta_connector.get_option_price("SPY", date, long_put_strike, "P")
        
        if short_put_price is None or long_put_price is None:
            return None
        
        # Calculate net credit
        net_credit = short_put_price - long_put_price
        
        if net_credit <= 0:
            return None
        
        # Calculate position size
        contracts = self.calculate_position_size("PUT_CREDIT_SPREAD", short_put_price, portfolio_value)
        
        # Calculate P&L
        premium_collected = net_credit * contracts * 100
        
        # Simulate outcome (75% success rate)
        is_profitable = np.random.random() < 0.75
        if is_profitable:
            pnl = premium_collected
        else:
            max_loss = (5 - net_credit) * contracts * 100
            pnl = -max_loss
        
        # Execute in backtrader
        if pnl > 0:
            self.buy(size=contracts)
        else:
            self.sell(size=abs(contracts))
        
        return {
            'date': date,
            'strategy': 'PUT_CREDIT_SPREAD',
            'spy_price': spy_price,
            'vix': vix,
            'strike': f"{short_put_strike}/{long_put_strike}",
            'right': 'P',
            'option_price': net_credit,
            'contracts': contracts,
            'premium': premium_collected,
            'pnl': pnl,
            'portfolio_value': portfolio_value
        }
    
    def execute_call_credit_spread(self, date: str, spy_price: float, vix: float, portfolio_value: float) -> Optional[Dict]:
        """Execute Call Credit Spread using real option data"""
        # Similar to put credit spread but with calls
        if not self.use_theta_data:
            return self.simulate_call_credit_spread(date, spy_price, vix, portfolio_value)
        
        short_call_strike = round(spy_price * 1.05)
        long_call_strike = short_call_strike + 5
        
        short_call_price = self.theta_connector.get_option_price("SPY", date, short_call_strike, "C")
        long_call_price = self.theta_connector.get_option_price("SPY", date, long_call_strike, "C")
        
        if short_call_price is None or long_call_price is None:
            return None
        
        net_credit = short_call_price - long_call_price
        
        if net_credit <= 0:
            return None
        
        contracts = self.calculate_position_size("CALL_CREDIT_SPREAD", short_call_price, portfolio_value)
        premium_collected = net_credit * contracts * 100
        
        is_profitable = np.random.random() < 0.75
        if is_profitable:
            pnl = premium_collected
        else:
            max_loss = (5 - net_credit) * contracts * 100
            pnl = -max_loss
        
        if pnl > 0:
            self.buy(size=contracts)
        else:
            self.sell(size=abs(contracts))
        
        return {
            'date': date,
            'strategy': 'CALL_CREDIT_SPREAD',
            'spy_price': spy_price,
            'vix': vix,
            'strike': f"{short_call_strike}/{long_call_strike}",
            'right': 'C',
            'option_price': net_credit,
            'contracts': contracts,
            'premium': premium_collected,
            'pnl': pnl,
            'portfolio_value': portfolio_value
        }
    
    def execute_iron_butterfly(self, date: str, spy_price: float, vix: float, portfolio_value: float) -> Optional[Dict]:
        """Execute Iron Butterfly using real option data"""
        if not self.use_theta_data:
            return self.simulate_iron_butterfly(date, spy_price, vix, portfolio_value)
        
        # ATM strikes
        atm_strike = round(spy_price)
        wing_strike_call = atm_strike + 10
        wing_strike_put = atm_strike - 10
        
        # Get prices
        short_call_price = self.theta_connector.get_option_price("SPY", date, atm_strike, "C")
        short_put_price = self.theta_connector.get_option_price("SPY", date, atm_strike, "P")
        long_call_price = self.theta_connector.get_option_price("SPY", date, wing_strike_call, "C")
        long_put_price = self.theta_connector.get_option_price("SPY", date, wing_strike_put, "P")
        
        if any(x is None for x in [short_call_price, short_put_price, long_call_price, long_put_price]):
            return None
        
        net_credit = safe_sum(short_call_price, short_put_price) - safe_sum(long_call_price, long_put_price)
        
        if net_credit <= 0:
            return None
        
        try:
            avg_option_price = (float(short_call_price) + float(short_put_price)) / 2
        except Exception:
            return None
        
        contracts = self.calculate_position_size("IRON_BUTTERFLY", avg_option_price, portfolio_value)
        premium_collected = net_credit * contracts * 100
        
        # Iron butterfly has lower success rate but higher premium
        is_profitable = np.random.random() < 0.60
        if is_profitable:
            pnl = premium_collected
        else:
            max_loss = (10 - net_credit) * contracts * 100
            pnl = -max_loss
        
        if pnl > 0:
            self.buy(size=contracts)
        else:
            self.sell(size=abs(contracts))
        
        return {
            'date': date,
            'strategy': 'IRON_BUTTERFLY',
            'spy_price': spy_price,
            'vix': vix,
            'strike': f"{atm_strike}",
            'right': 'IB',
            'option_price': net_credit,
            'contracts': contracts,
            'premium': premium_collected,
            'pnl': pnl,
            'portfolio_value': portfolio_value
        }
    
    def execute_diagonal_spread(self, date: str, spy_price: float, vix: float, portfolio_value: float) -> Optional[Dict]:
        """Execute Diagonal Spread (simplified as debit spread)"""
        if not self.use_theta_data:
            return self.simulate_diagonal_spread(date, spy_price, vix, portfolio_value)
        
        # For simplicity, implement as ATM call debit spread
        atm_strike = round(spy_price)
        otm_strike = atm_strike + 5
        
        # Get prices (assume we're buying ATM, selling OTM)
        long_call_price = self.theta_connector.get_option_price("SPY", date, atm_strike, "C")
        short_call_price = self.theta_connector.get_option_price("SPY", date, otm_strike, "C")
        
        if long_call_price is None or short_call_price is None:
            return None
        
        net_debit = long_call_price - short_call_price
        
        if net_debit <= 0:
            return None
        
        contracts = self.calculate_position_size("DIAGONAL", long_call_price, portfolio_value)
        cost = net_debit * contracts * 100
        
        # Diagonal spreads have moderate success rate
        is_profitable = np.random.random() < 0.65
        if is_profitable:
            # Profit is typically 2-3x the debit paid
            pnl = cost * 2.5
        else:
            # Loss is typically the full debit
            pnl = -cost
        
        if pnl > 0:
            self.buy(size=contracts)
        else:
            self.sell(size=abs(contracts))
        
        return {
            'date': date,
            'strategy': 'DIAGONAL',
            'spy_price': spy_price,
            'vix': vix,
            'strike': f"{atm_strike}/{otm_strike}",
            'right': 'C',
            'option_price': net_debit,
            'contracts': contracts,
            'premium': -cost,  # Negative because it's a debit
            'pnl': pnl,
            'portfolio_value': portfolio_value
        }
    
    # Simulation methods for when ThetaData is unavailable
    def simulate_iron_condor(self, date: str, spy_price: float, vix: float, portfolio_value: float) -> Optional[Dict]:
        """Simulate iron condor when real data unavailable"""
        net_credit = 0.50 + (vix - 15) * 0.05  # Higher VIX = higher credit
        contracts = max(1, int(portfolio_value * 0.02 / (net_credit * 100)))
        premium_collected = net_credit * contracts * 100
        
        is_profitable = np.random.random() < 0.80
        pnl = premium_collected if is_profitable else -premium_collected * 2
        
        if pnl > 0:
            self.buy(size=contracts)
        else:
            self.sell(size=abs(contracts))
        
        return {
            'date': date, 'strategy': 'IRON_CONDOR_SIM', 'spy_price': spy_price, 'vix': vix,
            'strike': 'SIMULATED', 'right': 'IC', 'option_price': net_credit, 'contracts': contracts,
            'premium': premium_collected, 'pnl': pnl, 'portfolio_value': portfolio_value
        }
    
    def simulate_put_credit_spread(self, date: str, spy_price: float, vix: float, portfolio_value: float) -> Optional[Dict]:
        """Simulate put credit spread"""
        net_credit = 0.30 + (vix - 15) * 0.03
        contracts = max(1, int(portfolio_value * 0.02 / (net_credit * 100)))
        premium_collected = net_credit * contracts * 100
        
        is_profitable = np.random.random() < 0.75
        pnl = premium_collected if is_profitable else -premium_collected * 1.5
        
        if pnl > 0:
            self.buy(size=contracts)
        else:
            self.sell(size=abs(contracts))
        
        return {
            'date': date, 'strategy': 'PUT_CREDIT_SPREAD_SIM', 'spy_price': spy_price, 'vix': vix,
            'strike': 'SIMULATED', 'right': 'P', 'option_price': net_credit, 'contracts': contracts,
            'premium': premium_collected, 'pnl': pnl, 'portfolio_value': portfolio_value
        }
    
    def simulate_call_credit_spread(self, date: str, spy_price: float, vix: float, portfolio_value: float) -> Optional[Dict]:
        """Simulate call credit spread"""
        return self.simulate_put_credit_spread(date, spy_price, vix, portfolio_value)  # Similar logic
    
    def simulate_iron_butterfly(self, date: str, spy_price: float, vix: float, portfolio_value: float) -> Optional[Dict]:
        """Simulate iron butterfly"""
        net_credit = 0.80 + (vix - 15) * 0.08
        contracts = max(1, int(portfolio_value * 0.02 / (net_credit * 100)))
        premium_collected = net_credit * contracts * 100
        
        is_profitable = np.random.random() < 0.60
        pnl = premium_collected if is_profitable else -premium_collected * 3
        
        if pnl > 0:
            self.buy(size=contracts)
        else:
            self.sell(size=abs(contracts))
        
        return {
            'date': date, 'strategy': 'IRON_BUTTERFLY_SIM', 'spy_price': spy_price, 'vix': vix,
            'strike': 'SIMULATED', 'right': 'IB', 'option_price': net_credit, 'contracts': contracts,
            'premium': premium_collected, 'pnl': pnl, 'portfolio_value': portfolio_value
        }
    
    def simulate_diagonal_spread(self, date: str, spy_price: float, vix: float, portfolio_value: float) -> Optional[Dict]:
        """Simulate diagonal spread"""
        net_debit = 2.50 - (vix - 15) * 0.10  # Lower VIX = higher cost
        contracts = max(1, int(portfolio_value * 0.02 / (net_debit * 100)))
        cost = net_debit * contracts * 100
        
        is_profitable = np.random.random() < 0.65
        pnl = cost * 2 if is_profitable else -cost
        
        if pnl > 0:
            self.buy(size=contracts)
        else:
            self.sell(size=abs(contracts))
        
        return {
            'date': date, 'strategy': 'DIAGONAL_SIM', 'spy_price': spy_price, 'vix': vix,
            'strike': 'SIMULATED', 'right': 'C', 'option_price': net_debit, 'contracts': contracts,
            'premium': -cost, 'pnl': pnl, 'portfolio_value': portfolio_value
        }
    
    def next(self):
        """
        Main strategy logic called on each bar
        """
        # Reset daily counters if new day
        current_date = self.datetime.date()
        if self.last_trade_date != current_date:
            self.risk_manager.reset_daily_counters()
            self.last_trade_date = current_date
        
        # Analyze market conditions
        strategy, conditions = self.analyze_market_conditions()
        
        # Execute strategy if conditions are met
        if strategy != "NO_TRADE":
            self.execute_strategy(strategy, conditions)
    
    def stop(self):
        """Called when backtest ends - print results"""
        final_value = self.broker.get_value()
        total_return = ((final_value - self.params.starting_capital) / self.params.starting_capital) * 100
        
        print("\n" + "=" * 80)
        print("🎯 THETADATA MULTI-REGIME BACKTRADER RESULTS")
        print("=" * 80)
        print(f"💰 Starting Capital: ${self.params.starting_capital:,.2f}")
        print(f"💰 Final Value: ${final_value:,.2f}")
        print(f"📊 Total Return: {total_return:.2f}%")
        print(f"💵 Profit/Loss: ${final_value - self.params.starting_capital:,.2f}")
        print(f"🎯 Total Trades: {len(self.trade_log)}")
        
        if self.trade_log:
            df = pd.DataFrame(self.trade_log)
            strategies = df['strategy'].value_counts()
            print(f"\n📊 Strategy Breakdown:")
            for strategy, count in strategies.items():
                strategy_pnl = df[df['strategy'] == strategy]['pnl'].sum()
                print(f"   {strategy}: {count} trades, ${strategy_pnl:,.2f} P&L")
            
            # Save detailed results
            df.to_csv(f'theta_multi_regime_backtrader_trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)
            print(f"\n💾 Detailed results saved to CSV file")
        
        print("=" * 80)

def run_theta_multi_regime_backtest(start_date: str = '2025-01-01', end_date: str = '2025-06-30'):
    """
    Run the ThetaData Multi-Regime Options Strategy Backtest
    """
    print("🚀 INITIALIZING THETADATA MULTI-REGIME BACKTRADER")
    print("=" * 80)
    
    # Initialize Cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(25000)
    cerebro.broker.setcommission(commission=0.65)  # $0.65 per options contract
    
    # Download market data
    print(f"📊 Downloading market data: {start_date} to {end_date}")
    
    try:
        # SPY data
        spy_df = yf.download('SPY', start=start_date, end=end_date)
        vix_df = yf.download('^VIX', start=start_date, end=end_date)
        
        # Handle MultiIndex columns
        if isinstance(spy_df.columns, pd.MultiIndex):
            spy_df.columns = spy_df.columns.get_level_values(0)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)
        
        # Clean and align data
        spy_df = spy_df.dropna()
        vix_df = vix_df.dropna()
        
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in spy_df.columns:
                spy_df[col] = spy_df['Close'] if col != 'Volume' else 1000000
            if col not in vix_df.columns:
                vix_df[col] = vix_df['Close'] if col != 'Volume' else 1000000
        
        # Select required columns
        spy_df = spy_df[required_cols]
        vix_df = vix_df[required_cols]
        
        # Fill any remaining NaN values
        spy_df = spy_df.fillna(method='ffill').fillna(method='bfill')
        vix_df = vix_df.fillna(method='ffill').fillna(method='bfill')
        
        print(f"✅ SPY data: {len(spy_df)} days")
        print(f"✅ VIX data: {len(vix_df)} days")
        
        # Add data feeds to Cerebro
        data_spy = bt.feeds.PandasData(dataname=spy_df)
        data_vix = bt.feeds.PandasData(dataname=vix_df)
        
        cerebro.adddata(data_spy, name='SPY')
        cerebro.adddata(data_vix, name='VIX')
        
        # Add strategy
        cerebro.addstrategy(MultiRegimeOptionsStrategy)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        print("🔄 Running backtest...")
        
        # Run backtest
        results = cerebro.run()
        strategy_result = results[0]
        
        # Print additional analytics
        print(f"\n📊 ADDITIONAL ANALYTICS:")
        print(f"📈 Sharpe Ratio: {strategy_result.analyzers.sharpe.get_analysis().get('sharperatio', 'N/A')}")
        print(f"📉 Max Drawdown: {strategy_result.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 'N/A'):.2f}%")
        
        # Plot results (optional)
        try:
            cerebro.plot(style='candlestick', barup='green', bardown='red')
        except Exception as e:
            print(f"Note: Plotting disabled due to: {e}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        return None

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run the backtest
    results = run_theta_multi_regime_backtest()
    
    if results:
        print("\n✅ Backtest completed successfully!")
    else:
        print("\n❌ Backtest failed!") 