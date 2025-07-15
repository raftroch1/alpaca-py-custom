#!/usr/bin/env python3
"""
BASE THETA STRATEGY TEMPLATE
A boilerplate template for all future option strategies using ThetaData.

This template provides:
- Proven ThetaData connection and API calls
- Standardized logging to separate folder
- Base strategy framework
- Error handling and data validation
- Performance tracking and CSV export

Usage:
    Inherit from BaseThetaStrategy and implement:
    - analyze_market_conditions()
    - execute_strategy()
    - calculate_position_size()
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
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')

# Add alpaca imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


class BaseThetaStrategy(ABC):
    """
    Base class for all ThetaData-based option strategies.
    
    Provides proven ThetaData connection, logging, and base functionality.
    All specific strategies should inherit from this class.
    """
    
    def __init__(self, 
                 strategy_name: str,
                 version: str = "v1",
                 starting_capital: float = 25000,
                 max_risk_per_trade: float = 0.03,
                 target_profit_per_trade: float = 0.002):
        """
        Initialize base strategy with proven ThetaData connection.
        
        Args:
            strategy_name: Name of the strategy (e.g., "vix_contrarian")
            version: Version string (e.g., "v1", "v2")
            starting_capital: Starting capital amount
            max_risk_per_trade: Maximum risk per trade as percentage
            target_profit_per_trade: Target profit per trade as percentage
        """
        self.strategy_name = strategy_name
        self.version = version
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.target_profit_per_trade = target_profit_per_trade
        
        # Strategy tracking
        self.trades = []
        self.skipped_trades = 0
        self.total_trades = 0
        
        # ThetaData connection (proven working format)
        self.theta_base_url = "http://127.0.0.1:25510"
        self.session = requests.Session()
        
        # Initialize logging and clients
        self.setup_logging()
        self.initialize_clients()
    
    def setup_logging(self):
        """Setup logging to strategies/logs folder with strategy name and version"""
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{self.strategy_name}_{self.version}_{timestamp}.log"
        log_path = os.path.join(log_dir, log_filename)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(f"{self.strategy_name}_{self.version}")
        self.logger.info(f"🚀 Initialized {self.strategy_name} {self.version}")
        self.logger.info(f"📁 Log file: {log_path}")
    
    def initialize_clients(self):
        """Initialize all data clients including corrected ThetaData"""
        self.logger.info("🔄 Initializing real data clients...")
        
        try:
            # 1. Test ThetaData connection with corrected format
            self.test_thetadata_connection()
            
            # 2. Initialize Alpaca client
            load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
            api_key = os.getenv('ALPACA_API_KEY')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            
            if not api_key or not secret_key:
                self.logger.warning("⚠️ Using paper trading keys for data access")
                api_key = "PK6UMC7TLJNKSTMJWX5V"
                secret_key = "aJEIv7ZnBhAzZpTGKJSGEXxhTNZJEUyZlNgTJSHW"
            
            self.alpaca_client = StockHistoricalDataClient(api_key, secret_key)
            self.logger.info("✅ Alpaca client initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Client initialization error: {e}")
            sys.exit(1)
    
    def test_thetadata_connection(self):
        """Test ThetaData connection with corrected API format"""
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
                data = response.json()
                if 'response' in data and data['response']:
                    self.logger.info("✅ ThetaData connection successful")
                    self.logger.info("✅ Options historical data accessible")
                    self.logger.info("✅ Subscription working correctly")
                    return True
                else:
                    self.logger.warning("⚠️ ThetaData connected but no data returned")
                    return False
            else:
                self.logger.error(f"❌ ThetaData connection failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ ThetaData connection error: {e}")
            return False
    
    def get_option_price(self, symbol: str, exp_date: str, strike: float, right: str, date: str) -> Optional[float]:
        """
        Get option price using corrected ThetaData API format.
        
        Args:
            symbol: Underlying symbol (e.g., 'SPY')
            exp_date: Expiration date in YYYYMMDD format
            strike: Strike price in dollars
            right: 'C' for call, 'P' for put
            date: Date in YYYYMMDD format
            
        Returns:
            Option price or None if not available
        """
        try:
            url = f"{self.theta_base_url}/v2/hist/option/eod"
            params = {
                'root': symbol,
                'exp': exp_date,
                'strike': int(strike * 1000),  # Convert to thousandths
                'right': right,
                'start_date': date,
                'end_date': date
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'response' in data and data['response']:
                    # Close price is at index 5
                    close_price = data['response'][0][5]
                    if close_price > 0:
                        return close_price
            
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Option price error for {symbol} {strike}{right}: {e}")
            return None
    
    def get_spy_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get SPY data using Alpaca client"""
        self.logger.info(f"📊 Fetching SPY data from {start_date} to {end_date}...")
        
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=["SPY"],
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=datetime.strptime(start_date, '%Y-%m-%d'),
                end=datetime.strptime(end_date, '%Y-%m-%d')
            )
            
            bars = self.alpaca_client.get_stock_bars(request_params)
            
            spy_data = []
            for bar in bars:
                spy_data.append({
                    'date': bar.timestamp.strftime('%Y-%m-%d'),
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume)
                })
            
            df = pd.DataFrame(spy_data)
            df['date'] = pd.to_datetime(df['date'])
            
            self.logger.info(f"✅ Retrieved {len(df)} SPY trading days")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ SPY data error: {e}")
            return None
    
    def get_vix_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Get VIX data using yfinance"""
        self.logger.info(f"📊 Fetching VIX data from {start_date} to {end_date}...")
        
        try:
            vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
            
            if vix.empty:
                self.logger.error("❌ No VIX data returned")
                return None
            
            vix_data = []
            for date, row in vix.iterrows():
                vix_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'vix_close': float(row['Close'])
                })
            
            df = pd.DataFrame(vix_data)
            df['date'] = pd.to_datetime(df['date'])
            
            self.logger.info(f"✅ Retrieved {len(df)} VIX trading days")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ VIX data error: {e}")
            return None
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log trade details and add to trades list"""
        self.trades.append(trade_data)
        self.total_trades += 1
        
        # Log key trade details
        strategy_type = trade_data.get('strategy', 'UNKNOWN')
        profit = trade_data.get('profit', 0)
        self.logger.info(f"📈 {strategy_type}: ${profit:.2f} | Account: ${self.current_capital:,.2f}")
    
    def skip_trade(self, reason: str, date: str):
        """Log skipped trade with reason"""
        self.skipped_trades += 1
        self.logger.info(f"⏭️  SKIPPED ({date}): {reason}")
    
    def save_results(self) -> str:
        """Save trading results to CSV file in logs folder"""
        if not self.trades:
            self.logger.warning("⚠️ No trades to save")
            return ""
        
        # Create results DataFrame
        df = pd.DataFrame(self.trades)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.strategy_name}_{self.version}_{timestamp}_trades.csv"
        
        # Save to logs folder
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        filepath = os.path.join(log_dir, filename)
        
        df.to_csv(filepath, index=False)
        self.logger.info(f"💾 Results saved to: {filename}")
        
        return filepath
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {}
        
        df = pd.DataFrame(self.trades)
        total_profit = df['profit'].sum()
        profitable_trades = len(df[df['profit'] > 0])
        total_trades = len(df)
        
        return {
            'total_return': (total_profit / self.starting_capital) * 100,
            'total_profit': total_profit,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': (profitable_trades / total_trades) * 100 if total_trades > 0 else 0,
            'avg_profit_per_trade': total_profit / total_trades if total_trades > 0 else 0,
            'skipped_trades': self.skipped_trades
        }
    
    def print_performance_summary(self):
        """Print comprehensive performance summary"""
        metrics = self.calculate_performance_metrics()
        
        if not metrics:
            self.logger.info("❌ No performance data available")
            return
        
        self.logger.info("=" * 70)
        self.logger.info(f"📊 {self.strategy_name.upper()} {self.version} RESULTS (REAL THETA DATA ONLY)")
        self.logger.info("=" * 70)
        self.logger.info(f"🎯 Total Trades: {metrics['total_trades']}")
        self.logger.info(f"⏭️  Skipped (No Data): {metrics['skipped_trades']}")
        self.logger.info(f"💰 Starting Capital: ${self.starting_capital:,.2f}")
        self.logger.info(f"💰 Final Capital: ${self.current_capital:,.2f}")
        self.logger.info(f"📊 Total Return: {metrics['total_return']:.2f}%")
        self.logger.info(f"💵 Total Profit: ${metrics['total_profit']:,.2f}")
        self.logger.info(f"✅ Win Rate: {metrics['win_rate']:.1f}%")
        self.logger.info(f"💸 Average Profit per Trade: ${metrics['avg_profit_per_trade']:.2f}")
        self.logger.info("")
        self.logger.info("🔧 Method: REAL THETA DATA ONLY (no simulation fallback)")
    
    @abstractmethod
    def analyze_market_conditions(self, spy_price: float, vix_level: float, date: str) -> Dict[str, Any]:
        """
        Analyze market conditions and determine strategy parameters.
        
        Args:
            spy_price: Current SPY price
            vix_level: Current VIX level
            date: Current date string
            
        Returns:
            Dictionary with strategy decisions and parameters
        """
        pass
    
    @abstractmethod
    def execute_strategy(self, market_analysis: Dict[str, Any], spy_price: float, date: str) -> Optional[Dict[str, Any]]:
        """
        Execute the strategy based on market analysis.
        
        Args:
            market_analysis: Results from analyze_market_conditions()
            spy_price: Current SPY price
            date: Current date string
            
        Returns:
            Trade data dictionary or None if trade skipped
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, strategy_type: str, premium_collected: float) -> int:
        """
        Calculate position size based on strategy and risk management.
        
        Args:
            strategy_type: Type of strategy being executed
            premium_collected: Premium collected per contract
            
        Returns:
            Number of contracts to trade
        """
        pass
    
    def run_backtest(self, start_date: str, end_date: str):
        """
        Run the strategy backtest.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        self.logger.info(f"🚀 STARTING {self.strategy_name.upper()} {self.version} BACKTEST")
        self.logger.info("🔧 USING CORRECTED THETA DATA API (NO SIMULATION)")
        self.logger.info(f"📅 Period: {start_date} to {end_date}")
        self.logger.info("=" * 70)
        
        # Get market data
        spy_data = self.get_spy_data(start_date, end_date)
        vix_data = self.get_vix_data(start_date, end_date)
        
        if spy_data is None or vix_data is None:
            self.logger.error("❌ Failed to get market data")
            return
        
        # Merge data
        market_data = pd.merge(spy_data, vix_data, on='date', how='inner')
        self.logger.info(f"📊 Market data: {len(market_data)} trading days")
        
        # Run strategy for each trading day
        for _, row in market_data.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            spy_price = row['close']
            vix_level = row['vix_close']
            
            # Analyze market conditions
            market_analysis = self.analyze_market_conditions(spy_price, vix_level, date_str)
            
            # Execute strategy
            trade_result = self.execute_strategy(market_analysis, spy_price, date_str)
            
            if trade_result:
                self.log_trade(trade_result)
                self.current_capital += trade_result['profit']
        
        # Print final results
        self.print_performance_summary()
        
        # Save results
        self.save_results() 