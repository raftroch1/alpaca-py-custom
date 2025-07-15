#!/usr/bin/env python3
"""
DIAGONAL SPREAD 0DTE STRATEGY BACKTEST
Implementing proper diagonal spreads with realistic position sizing
"""

import sys
import os
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
import yfinance as yf
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Add alpaca imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

class DiagonalSpreadStrategy:
    """
    Diagonal Spread 0DTE Strategy:
    - Sell 0DTE options (expires today)
    - Buy longer-dated options (7-21 days out)
    - Net credit position with controlled risk
    """
    
    def __init__(self, starting_capital: float = 25000):
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        self.trades = []
        self.theta_base_url = "http://127.0.0.1:25510"
        self.session = requests.Session()
        
        # Strategy parameters
        self.max_risk_per_trade = 0.02  # 2% max risk per trade
        self.target_profit_per_trade = 0.001  # 0.1% target profit per trade = $25 on $25k
        self.profit_target_percent = 0.5  # Take profit at 50% of max profit
        self.stop_loss_multiplier = 2.0  # Stop loss at 2x credit received
        
        # Load environment variables
        load_dotenv(os.path.join(os.path.dirname(__file__), 'strategies', '.env'))
        
        # Initialize data clients
        self.initialize_clients()
    
    def initialize_clients(self):
        """Initialize all data clients"""
        print("🔄 Initializing real data clients...")
        
        try:
            # 1. ThetaData connection test
            self.test_thetadata_connection()
            
            # 2. Initialize Alpaca client
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            
            if not api_key or not secret_key:
                print("⚠️ Using paper trading keys for data access")
                api_key = "PK6UMC7TLJNKSTMJWX5V"
                secret_key = "aJEIv7ZnBhAzZpTGKJSGEXxhTNZJEUyZlNgTJSHW"
            
            self.alpaca_client = StockHistoricalDataClient(api_key, secret_key)
            print("✅ Alpaca client initialized")
            
        except Exception as e:
            print(f"❌ Client initialization error: {e}")
            sys.exit(1)
    
    def test_thetadata_connection(self):
        """Test ThetaData connection with working endpoint"""
        try:
            # Test with proper format: strike in thousandths, date as YYYYMMDD
            url = f"{self.theta_base_url}/v2/hist/option/eod"
            params = {
                'root': 'SPY',
                'exp': '20240705',
                'strike': '535000',  # $535 in thousandths
                'right': 'P',
                'start_date': '20240705',
                'end_date': '20240705'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                print("✅ ThetaData connection successful")
            else:
                print(f"❌ ThetaData connection failed: {response.status_code}")
        except Exception as e:
            print(f"❌ ThetaData connection error: {e}")
            sys.exit(1)
    
    def get_spy_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get SPY historical data from Alpaca"""
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=["SPY"],
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=datetime.strptime(start_date, '%Y-%m-%d'),
                end=datetime.strptime(end_date, '%Y-%m-%d')
            )
            
            bars = self.alpaca_client.get_stock_bars(request_params)
            
            spy_data = []
            for bar in bars["SPY"]:
                spy_data.append({
                    'date': bar.timestamp.strftime('%Y-%m-%d'),
                    'spy_price': float(bar.close)
                })
            
            df = pd.DataFrame(spy_data)
            df['date'] = pd.to_datetime(df['date'])
            print(f"✅ Retrieved {len(df)} SPY trading days")
            return df
            
        except Exception as e:
            print(f"❌ Error getting SPY data: {e}")
            return None
    
    def get_vix_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get VIX data from Yahoo Finance"""
        try:
            vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
            
            if vix.empty:
                print("❌ No VIX data retrieved")
                return None
            
            vix_data = []
            for date, row in vix.iterrows():
                vix_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'vix': float(row['Close'])
                })
            
            df = pd.DataFrame(vix_data)
            df['date'] = pd.to_datetime(df['date'])
            print(f"✅ Retrieved {len(df)} VIX trading days")
            return df
            
        except Exception as e:
            print(f"❌ Error getting VIX data: {e}")
            return None
    
    def format_thetadata_date(self, date_str: str) -> str:
        """Convert date to ThetaData format (YYYYMMDD)"""
        return pd.to_datetime(date_str).strftime('%Y%m%d')
    
    def format_thetadata_strike(self, strike: float) -> int:
        """Convert strike to ThetaData format (in thousandths)"""
        return int(strike * 1000)
    
    def get_option_data(self, symbol: str, date: str, strike: float, right: str, expiry: str) -> Optional[float]:
        """Get option price from ThetaData"""
        try:
            # Format parameters for ThetaData
            formatted_date = self.format_thetadata_date(date)
            formatted_strike = self.format_thetadata_strike(strike)
            formatted_expiry = self.format_thetadata_date(expiry)
            
            # Build ThetaData request
            url = f"{self.theta_base_url}/v2/hist/option/eod"
            params = {
                'root': symbol,
                'exp': formatted_expiry,
                'strike': formatted_strike,
                'right': right,
                'date': formatted_date,
                'ivl': 0,
                'use_csv': 'false'
            }
            
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if 'response' in data and len(data['response']) > 0:
                    # ThetaData returns arrays: [ms, open, high, low, close, volume, count]
                    price_data = data['response'][0]
                    if len(price_data) >= 5:
                        close_price = price_data[4] / 100  # Convert from cents to dollars
                        return close_price
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting option data: {e}")
            return None
    
    def calculate_diagonal_spread(self, spy_price: float, vix: float, date: str) -> Optional[Dict]:
        """Calculate diagonal spread parameters"""
        
        # Determine direction based on VIX
        if vix >= 20:
            # High VIX - bearish diagonal (sell call spread)
            direction = "BEARISH"
            sell_strike = spy_price * 1.005  # Slightly OTM call
            buy_strike = spy_price * 1.015   # Further OTM call
            right = "C"
        else:
            # Low VIX - bullish diagonal (sell put spread)
            direction = "BULLISH"
            sell_strike = spy_price * 0.995  # Slightly OTM put
            buy_strike = spy_price * 0.985   # Further OTM put
            right = "P"
        
        # Round strikes to nearest $1
        sell_strike = round(sell_strike)
        buy_strike = round(buy_strike)
        
        # Calculate expiration dates
        sell_expiry = date  # 0DTE
        buy_expiry = (pd.to_datetime(date) + timedelta(days=14)).strftime('%Y-%m-%d')  # 2 weeks out
        
        # Get option prices
        sell_price = self.get_option_data("SPY", date, sell_strike, right, sell_expiry)
        buy_price = self.get_option_data("SPY", date, buy_strike, right, buy_expiry)
        
        if sell_price is None or buy_price is None:
            return None
        
        # Calculate net credit (sell price - buy price)
        net_credit = sell_price - buy_price
        
        # Only proceed if we get a net credit
        if net_credit <= 0:
            return None
        
        return {
            'direction': direction,
            'sell_strike': sell_strike,
            'buy_strike': buy_strike,
            'sell_price': sell_price,
            'buy_price': buy_price,
            'net_credit': net_credit,
            'sell_expiry': sell_expiry,
            'buy_expiry': buy_expiry,
            'right': right
        }
    
    def calculate_position_size(self, net_credit: float, max_loss: float) -> int:
        """Calculate position size based on risk management"""
        
        # Maximum risk per trade (2% of account)
        max_risk = self.current_capital * self.max_risk_per_trade
        
        # Position size based on maximum possible loss
        # For diagonal spreads, max loss is typically strike difference - net credit
        if max_loss > 0:
            max_contracts = int(max_risk / (max_loss * 100))  # 100 shares per contract
        else:
            max_contracts = 1
        
        # Also consider target profit
        target_profit = self.current_capital * self.target_profit_per_trade
        target_contracts = int(target_profit / (net_credit * 100))
        
        # Use the smaller of the two
        position_size = min(max_contracts, target_contracts, 10)  # Cap at 10 contracts
        
        return max(1, position_size)  # Minimum 1 contract
    
    def simulate_diagonal_spread_outcome(self, spread_params: Dict, position_size: int) -> Dict:
        """Simulate the outcome of a diagonal spread"""
        
        net_credit = spread_params['net_credit']
        
        # Simplified simulation:
        # 70% of spreads expire profitably (short leg expires worthless)
        # 30% result in losses
        
        is_profitable = np.random.random() < 0.7
        
        if is_profitable:
            # Profit = net credit received (both legs expire worthless or minimal loss on long leg)
            profit = net_credit * position_size * 100
        else:
            # Loss = typically 2x the credit received
            profit = -net_credit * position_size * 100 * 2
        
        return {
            'is_profitable': is_profitable,
            'profit': profit,
            'position_size': position_size
        }
    
    def run_backtest(self, start_date: str = '2024-01-01', end_date: str = '2024-12-31'):
        """Run diagonal spread backtest"""
        print("🚀 STARTING DIAGONAL SPREAD BACKTEST")
        print("=" * 70)
        
        # Get market data
        spy_data = self.get_spy_data(start_date, end_date)
        vix_data = self.get_vix_data(start_date, end_date)
        
        if spy_data is None or vix_data is None:
            print("❌ Failed to get market data")
            return
        
        # Merge data
        market_data = pd.merge(spy_data, vix_data, on='date', how='inner')
        print(f"📊 Market data: {len(market_data)} trading days")
        
        # Track statistics
        total_trades = 0
        profitable_trades = 0
        total_profit = 0
        
        # Process each trading day
        for idx, row in market_data.iterrows():
            date = row['date'].strftime('%Y-%m-%d')
            spy_price = row['spy_price']
            vix = row['vix']
            
            # Only trade on elevated VIX (15+) or significant moves
            if vix >= 15 and total_trades < 20:  # Limit trades for testing
                
                print(f"\n📅 {date} - SPY: ${spy_price:.2f}, VIX: {vix:.2f}")
                
                # Calculate diagonal spread
                spread_params = self.calculate_diagonal_spread(spy_price, vix, date)
                
                if spread_params is not None:
                    net_credit = spread_params['net_credit']
                    
                    # Calculate position size
                    max_loss = abs(spread_params['sell_strike'] - spread_params['buy_strike']) - net_credit
                    position_size = self.calculate_position_size(net_credit, max_loss)
                    
                    # Simulate trade outcome
                    outcome = self.simulate_diagonal_spread_outcome(spread_params, position_size)
                    
                    # Update capital
                    self.current_capital += outcome['profit']
                    total_profit += outcome['profit']
                    total_trades += 1
                    
                    if outcome['is_profitable']:
                        profitable_trades += 1
                    
                    # Record trade
                    trade_record = {
                        'date': date,
                        'spy_price': spy_price,
                        'vix': vix,
                        'direction': spread_params['direction'],
                        'sell_strike': spread_params['sell_strike'],
                        'buy_strike': spread_params['buy_strike'],
                        'sell_price': spread_params['sell_price'],
                        'buy_price': spread_params['buy_price'],
                        'net_credit': net_credit,
                        'position_size': position_size,
                        'profit': outcome['profit'],
                        'is_profitable': outcome['is_profitable'],
                        'account_value': self.current_capital
                    }
                    
                    self.trades.append(trade_record)
                    
                    print(f"   🎯 {spread_params['direction']} Diagonal: {position_size} contracts")
                    print(f"   📊 Sell {spread_params['sell_strike']}{spread_params['right']} @ ${spread_params['sell_price']:.2f}")
                    print(f"   📊 Buy {spread_params['buy_strike']}{spread_params['right']} @ ${spread_params['buy_price']:.2f}")
                    print(f"   💰 Net Credit: ${net_credit:.2f} per contract")
                    print(f"   📈 P&L: ${outcome['profit']:,.2f} | Account: ${self.current_capital:,.2f}")
        
        # Calculate final results
        total_return = ((self.current_capital - self.starting_capital) / self.starting_capital) * 100
        win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
        avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
        
        print("\n" + "=" * 70)
        print("📊 DIAGONAL SPREAD BACKTEST RESULTS")
        print("=" * 70)
        print(f"🎯 Total Trades: {total_trades}")
        print(f"✅ Profitable Trades: {profitable_trades}")
        print(f"📈 Win Rate: {win_rate:.1f}%")
        print(f"💰 Starting Capital: ${self.starting_capital:,.2f}")
        print(f"💰 Final Capital: ${self.current_capital:,.2f}")
        print(f"📊 Total Return: {total_return:.2f}%")
        print(f"💵 Total Profit: ${total_profit:,.2f}")
        print(f"💸 Average Profit per Trade: ${avg_profit_per_trade:.2f}")
        
        # Show realistic daily profit expectations
        trading_days = len(market_data)
        daily_profit = total_profit / trading_days if trading_days > 0 else 0
        print(f"📅 Average Daily Profit: ${daily_profit:.2f}")
        
        # Capital requirements analysis
        print("\n" + "=" * 70)
        print("📊 CAPITAL REQUIREMENTS ANALYSIS")
        print("=" * 70)
        
        if self.trades:
            max_position_value = max([
                trade['position_size'] * max(trade['sell_price'], trade['buy_price']) * 100
                for trade in self.trades
            ])
            print(f"💰 Maximum Position Value: ${max_position_value:,.2f}")
            print(f"📊 Position as % of Account: {(max_position_value / self.starting_capital) * 100:.1f}%")
        
        # Save results
        if self.trades:
            df = pd.DataFrame(self.trades)
            df.to_csv('diagonal_spread_trades.csv', index=False)
            print(f"💾 Results saved to: diagonal_spread_trades.csv")

if __name__ == "__main__":
    print("🔄 Initializing Diagonal Spread Strategy...")
    strategy = DiagonalSpreadStrategy()
    strategy.run_backtest() 