#!/usr/bin/env python3
"""
LIVE HIGH CONVICTION 0DTE OPTIONS TRADER

Live paper trading implementation of the High Conviction strategy.
Uses Alpaca's paper trading API to execute real trades based on our backtested strategy.

Key Features:
1. Real-time market data and VIX monitoring
2. High conviction setup identification
3. Automated diagonal spread execution
4. Position monitoring and management
5. Risk management and stop losses
6. Performance tracking and logging

Requirements:
- Alpaca paper trading account
- API keys set as environment variables
- Market hours operation (9:30 AM - 4:00 PM ET)
"""

import os
import time
import logging
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    MarketOrderRequest,
    OptionLegRequest,
    LimitOrderRequest,
    ClosePositionRequest,
)
from alpaca.trading.enums import (
    AssetStatus,
    OrderSide,
    OrderClass,
    OrderType,
    TimeInForce,
    ContractType,
    ExerciseStyle,
)
from alpaca.data.requests import StockLatestTradeRequest

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LiveMarketDataFeed:
    """Real-time market data provider using Yahoo Finance and Alpaca."""
    
    def __init__(self, stock_client: StockHistoricalDataClient):
        self.stock_client = stock_client
        self.vix_history = []
        self.spy_history = []
        self.timestamps = []
        
    def get_current_vix(self) -> float:
        """Get current VIX value from Yahoo Finance."""
        try:
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period="1d", interval="1m")
            if not vix_data.empty:
                current_vix = float(vix_data['Close'].iloc[-1])
                logger.debug(f"Current VIX: {current_vix}")
                return current_vix
            else:
                logger.warning("No VIX data available, using fallback")
                return 15.0  # Fallback value
        except Exception as e:
            logger.error(f"Error fetching VIX: {e}")
            return 15.0  # Fallback value
    
    def get_current_spy_price(self) -> float:
        """Get current SPY price from Alpaca."""
        try:
            request = StockLatestTradeRequest(symbol_or_symbols="SPY")
            response = self.stock_client.get_stock_latest_trade(request)
            spy_price = float(response["SPY"].price)
            logger.debug(f"Current SPY: {spy_price}")
            return spy_price
        except Exception as e:
            logger.error(f"Error fetching SPY price: {e}")
            # Fallback to Yahoo Finance
            try:
                spy_ticker = yf.Ticker("SPY")
                spy_data = spy_ticker.history(period="1d", interval="1m")
                if not spy_data.empty:
                    return float(spy_data['Close'].iloc[-1])
            except:
                pass
            return 550.0  # Fallback value
    
    def update_market_data(self):
        """Update market data history."""
        current_time = datetime.now(ZoneInfo("America/New_York"))
        vix = self.get_current_vix()
        spy_price = self.get_current_spy_price()
        
        self.timestamps.append(current_time)
        self.vix_history.append(vix)
        self.spy_history.append(spy_price)
        
        # Keep only last 10 data points for analysis
        if len(self.vix_history) > 10:
            self.vix_history.pop(0)
            self.spy_history.pop(0)
            self.timestamps.pop(0)
        
        return vix, spy_price

class LiveConvictionAnalyzer:
    """Live implementation of high conviction market analysis."""
    
    def __init__(self):
        self.optimal_vix_range = (10.5, 16.0)
        self.min_conviction_score = 5
        
    def analyze_market_conditions(self, market_feed: LiveMarketDataFeed) -> Tuple[bool, Dict]:
        """Analyze current market conditions for high conviction setups."""
        
        if len(market_feed.vix_history) < 3:
            return False, {'conviction_score': 0, 'reason': 'Insufficient data'}
        
        current_vix = market_feed.vix_history[-1]
        current_spy = market_feed.spy_history[-1]
        
        conditions = {
            'vix_optimal': False,
            'momentum_favorable': False,
            'vix_stable': False,
            'conviction_score': 0,
            'recommended_size': 0,
            'current_vix': current_vix,
            'current_spy': current_spy,
            'analysis_time': datetime.now(ZoneInfo("America/New_York"))
        }
        
        # 1. VIX Analysis
        if self.optimal_vix_range[0] <= current_vix <= self.optimal_vix_range[1]:
            conditions['vix_optimal'] = True
            conditions['conviction_score'] += 3
            
            # Extra points for very low VIX
            if current_vix <= 13.0:
                conditions['conviction_score'] += 2
                
        # 2. VIX Stability
        if len(market_feed.vix_history) >= 3:
            recent_vix = market_feed.vix_history[-3:]
            vix_range = max(recent_vix) - min(recent_vix)
            vix_trend = recent_vix[-1] - recent_vix[0]
            
            if vix_range < 1.0:
                conditions['vix_stable'] = True
                conditions['conviction_score'] += 2
            elif vix_trend < -1.0:  # VIX declining
                conditions['conviction_score'] += 3
        
        # 3. SPY Momentum Analysis
        if len(market_feed.spy_history) >= 5:
            recent_spy = market_feed.spy_history[-5:]
            short_momentum = (recent_spy[-1] - recent_spy[-2]) / recent_spy[-2]
            med_momentum = (recent_spy[-1] - recent_spy[-5]) / recent_spy[-5]
            
            momentum_strength = abs(short_momentum) + abs(med_momentum)
            if 0.001 <= momentum_strength <= 0.02:  # Controlled movement
                conditions['momentum_favorable'] = True
                conditions['conviction_score'] += 2
        
        # 4. Time-based factors
        current_time = datetime.now(ZoneInfo("America/New_York"))
        if current_time.weekday() in [1, 2, 3]:  # Tue, Wed, Thu
            conditions['conviction_score'] += 1
        
        # Determine position size based on conviction
        if conditions['conviction_score'] >= 7:
            conditions['recommended_size'] = 200  # High conviction
        elif conditions['conviction_score'] >= 5:
            conditions['recommended_size'] = 150  # Medium conviction
        elif conditions['conviction_score'] >= 3:
            conditions['recommended_size'] = 100  # Low conviction
        else:
            conditions['recommended_size'] = 0    # Skip trade
        
        is_high_conviction = conditions['conviction_score'] >= self.min_conviction_score
        
        logger.info(f"Market Analysis - VIX: {current_vix:.2f}, SPY: {current_spy:.2f}, "
                   f"Conviction Score: {conditions['conviction_score']}, "
                   f"High Conviction: {is_high_conviction}")
        
        return is_high_conviction, conditions

class LiveOptionsTrader:
    """Live options trading execution using Alpaca API."""
    
    def __init__(self, trading_client: TradingClient):
        self.trading_client = trading_client
        self.active_positions = {}
        self.trade_history = []
        
    def find_optimal_diagonal_spread(self, spy_price: float, conviction_score: int) -> Optional[Dict]:
        """Find optimal diagonal spread options for current market conditions."""
        
        try:
            # Get today's date for 0DTE options
            today = datetime.now(ZoneInfo("America/New_York")).date()
            
            # Calculate strike range (around current SPY price)
            strike_range = 0.05  # 5% range
            min_strike = spy_price * (1 - strike_range)
            max_strike = spy_price * (1 + strike_range)
            
            # Get PUT options expiring today
            req = GetOptionContractsRequest(
                underlying_symbols=["SPY"],
                status=AssetStatus.ACTIVE,
                expiration_date=today,
                root_symbol="SPY",
                type=ContractType.PUT,
                style=ExerciseStyle.AMERICAN,
                strike_price_gte=str(min_strike),
                strike_price_lte=str(max_strike),
                limit=50
            )
            
            put_contracts = self.trading_client.get_option_contracts(req).option_contracts
            
            if len(put_contracts) < 2:
                logger.warning("Insufficient PUT options found for diagonal spread")
                return None
            
            # For diagonal spread, we want:
            # - Short PUT (higher strike, closer to money)
            # - Long PUT (lower strike, further from money)
            
            # Find optimal strikes
            short_put = None
            long_put = None
            
            # Look for puts around 5-10 points OTM
            target_short_strike = spy_price - 5
            target_long_strike = spy_price - 15
            
            # Find closest strikes
            for contract in put_contracts:
                strike = float(contract.strike_price)
                
                if not short_put or abs(strike - target_short_strike) < abs(float(short_put.strike_price) - target_short_strike):
                    if strike < spy_price:  # OTM put
                        short_put = contract
                
                if not long_put or abs(strike - target_long_strike) < abs(float(long_put.strike_price) - target_long_strike):
                    if strike < spy_price - 5:  # Further OTM
                        long_put = contract
            
            if not short_put or not long_put:
                logger.warning("Could not find suitable PUT contracts for diagonal spread")
                return None
            
            return {
                'strategy': 'diagonal_spread',
                'short_put': short_put,
                'long_put': long_put,
                'short_strike': float(short_put.strike_price),
                'long_strike': float(long_put.strike_price),
                'conviction_score': conviction_score,
                'spy_price': spy_price,
                'expiration': today
            }
            
        except Exception as e:
            logger.error(f"Error finding diagonal spread: {e}")
            return None
    
    def calculate_position_size(self, setup: Dict, account_value: float, conviction_score: int) -> int:
        """Calculate position size based on account value and conviction."""
        
        # Base risk parameters
        max_risk_per_trade = min(1500.0, account_value * 0.06)  # 6% max risk
        max_contracts = 250
        
        # Adjust based on conviction
        if conviction_score >= 7:
            recommended_size = 200
        elif conviction_score >= 5:
            recommended_size = 150
        else:
            recommended_size = 100
        
        # Account multiplier based on performance
        account_multiplier = min(2.0, account_value / 25000.0)
        adjusted_size = int(recommended_size * account_multiplier)
        
        # Risk-based validation (estimate max loss at $5 per contract)
        estimated_max_loss = 5.0
        max_contracts_by_risk = int(max_risk_per_trade / estimated_max_loss)
        
        final_size = min(adjusted_size, max_contracts_by_risk, max_contracts)
        final_size = max(10, final_size)  # Minimum meaningful size
        
        logger.info(f"Position sizing - Conviction: {conviction_score}, "
                   f"Account: ${account_value:.2f}, Size: {final_size}")
        
        return final_size
    
    def execute_diagonal_spread(self, setup: Dict, contracts: int) -> Optional[Dict]:
        """Execute diagonal spread trade."""
        
        try:
            # Create option legs for diagonal spread
            legs = [
                # Short PUT (sell for credit)
                OptionLegRequest(
                    symbol=setup['short_put'].symbol,
                    side=OrderSide.SELL,
                    ratio_qty=1
                ),
                # Long PUT (buy for protection)
                OptionLegRequest(
                    symbol=setup['long_put'].symbol,
                    side=OrderSide.BUY,
                    ratio_qty=1
                )
            ]
            
            # Create market order for immediate execution
            order_request = MarketOrderRequest(
                qty=contracts,
                order_class=OrderClass.MLEG,
                time_in_force=TimeInForce.DAY,
                legs=legs
            )
            
            # Submit order
            order_response = self.trading_client.submit_order(order_request)
            
            trade_info = {
                'order_id': order_response.id,
                'client_order_id': order_response.client_order_id,
                'strategy': setup['strategy'],
                'contracts': contracts,
                'short_put_symbol': setup['short_put'].symbol,
                'long_put_symbol': setup['long_put'].symbol,
                'short_strike': setup['short_strike'],
                'long_strike': setup['long_strike'],
                'spy_price': setup['spy_price'],
                'conviction_score': setup['conviction_score'],
                'timestamp': datetime.now(ZoneInfo("America/New_York")),
                'status': 'submitted'
            }
            
            # Store active position
            self.active_positions[order_response.id] = trade_info
            
            logger.info(f"TRADE EXECUTED - {setup['strategy']} x{contracts} - "
                       f"Short PUT {setup['short_strike']} / Long PUT {setup['long_strike']} - "
                       f"Order ID: {order_response.id}")
            
            return trade_info
            
        except Exception as e:
            logger.error(f"Error executing diagonal spread: {e}")
            return None
    
    def monitor_positions(self):
        """Monitor active positions for profit/loss management."""
        
        if not self.active_positions:
            return
        
        try:
            # Get current positions
            positions = self.trading_client.get_all_positions()
            
            for order_id, trade_info in list(self.active_positions.items()):
                try:
                    # Check order status
                    order = self.trading_client.get_order_by_id(order_id)
                    
                    if order.status in ['filled', 'partially_filled']:
                        trade_info['status'] = 'active'
                        
                        # Calculate current P&L
                        current_pnl = self.calculate_position_pnl(trade_info, positions)
                        trade_info['current_pnl'] = current_pnl
                        
                        # Check exit conditions
                        if self.should_exit_position(trade_info):
                            self.close_position(trade_info)
                    
                    elif order.status in ['canceled', 'rejected']:
                        logger.warning(f"Order {order_id} was {order.status}")
                        del self.active_positions[order_id]
                    
                except Exception as e:
                    logger.error(f"Error monitoring position {order_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in position monitoring: {e}")
    
    def calculate_position_pnl(self, trade_info: Dict, positions) -> float:
        """Calculate current P&L for a position."""
        
        try:
            total_pnl = 0.0
            
            # Find positions matching our trade
            for position in positions:
                if (position.symbol == trade_info['short_put_symbol'] or 
                    position.symbol == trade_info['long_put_symbol']):
                    
                    # Add unrealized P&L
                    total_pnl += float(position.unrealized_pl)
            
            return total_pnl
            
        except Exception as e:
            logger.error(f"Error calculating P&L: {e}")
            return 0.0
    
    def should_exit_position(self, trade_info: Dict) -> bool:
        """Determine if position should be closed."""
        
        current_time = datetime.now(ZoneInfo("America/New_York"))
        
        # Exit near market close (3:45 PM ET)
        if current_time.hour >= 15 and current_time.minute >= 45:
            logger.info(f"Closing position {trade_info['order_id']} - Market close approaching")
            return True
        
        # Exit if significant profit (>$1000 per contract)
        if 'current_pnl' in trade_info:
            profit_per_contract = trade_info['current_pnl'] / trade_info['contracts']
            
            if profit_per_contract > 20.0:  # $20+ per contract
                logger.info(f"Closing position {trade_info['order_id']} - Profit target hit: ${profit_per_contract:.2f}/contract")
                return True
            
            # Stop loss (-$8 per contract)
            if profit_per_contract < -8.0:
                logger.info(f"Closing position {trade_info['order_id']} - Stop loss hit: ${profit_per_contract:.2f}/contract")
                return True
        
        return False
    
    def close_position(self, trade_info: Dict):
        """Close an active position."""
        
        try:
            # Close short PUT position
            self.trading_client.close_position(
                symbol_or_asset_id=trade_info['short_put_symbol'],
                close_options=ClosePositionRequest(qty=str(trade_info['contracts']))
            )
            
            # Close long PUT position
            self.trading_client.close_position(
                symbol_or_asset_id=trade_info['long_put_symbol'],
                close_options=ClosePositionRequest(qty=str(trade_info['contracts']))
            )
            
            # Move to trade history
            trade_info['status'] = 'closed'
            trade_info['close_time'] = datetime.now(ZoneInfo("America/New_York"))
            self.trade_history.append(trade_info)
            
            # Remove from active positions
            if trade_info['order_id'] in self.active_positions:
                del self.active_positions[trade_info['order_id']]
            
            logger.info(f"Position closed - Order ID: {trade_info['order_id']}, "
                       f"Final P&L: ${trade_info.get('current_pnl', 0):.2f}")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")

class LiveTradingBot:
    """Main live trading bot orchestrating the High Conviction strategy."""
    
    def __init__(self):
        # Initialize Alpaca clients
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=True  # PAPER TRADING ONLY
        )
        
        self.stock_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key
        )
        
        # Initialize components
        self.market_feed = LiveMarketDataFeed(self.stock_client)
        self.analyzer = LiveConvictionAnalyzer()
        self.trader = LiveOptionsTrader(self.trading_client)
        
        # Trading state
        self.is_running = False
        self.last_trade_time = None
        self.daily_trades = 0
        self.max_daily_trades = 3
        
        # Performance tracking
        self.session_start_time = datetime.now(ZoneInfo("America/New_York"))
        self.starting_account_value = None
        
    def is_market_hours(self) -> bool:
        """Check if market is open for trading."""
        
        current_time = datetime.now(ZoneInfo("America/New_York"))
        
        # Skip weekends
        if current_time.weekday() >= 5:
            return False
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= current_time <= market_close
    
    def get_account_info(self) -> Dict:
        """Get current account information."""
        
        try:
            account = self.trading_client.get_account()
            return {
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'options_buying_power': float(account.options_buying_power) if hasattr(account, 'options_buying_power') else 0,
                'day_trade_count': int(account.daytrade_count) if hasattr(account, 'daytrade_count') else 0,
                'options_trading_level': int(account.options_trading_level) if hasattr(account, 'options_trading_level') else 0
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def log_daily_summary(self):
        """Log daily trading summary."""
        
        account_info = self.get_account_info()
        current_value = account_info.get('equity', 0)
        
        if self.starting_account_value:
            daily_pnl = current_value - self.starting_account_value
            daily_return = (daily_pnl / self.starting_account_value) * 100
        else:
            daily_pnl = 0
            daily_return = 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"DAILY TRADING SUMMARY - {datetime.now().strftime('%Y-%m-%d')}")
        logger.info(f"{'='*60}")
        logger.info(f"Starting Account Value: ${self.starting_account_value:,.2f}")
        logger.info(f"Current Account Value: ${current_value:,.2f}")
        logger.info(f"Daily P&L: ${daily_pnl:,.2f}")
        logger.info(f"Daily Return: {daily_return:.2f}%")
        logger.info(f"Trades Executed: {self.daily_trades}")
        logger.info(f"Active Positions: {len(self.trader.active_positions)}")
        logger.info(f"{'='*60}")
    
    def run_trading_session(self):
        """Run a complete trading session."""
        
        logger.info("Starting Live High Conviction Trading Bot")
        
        # Get starting account value
        account_info = self.get_account_info()
        self.starting_account_value = account_info.get('equity', 25000)
        
        logger.info(f"Account Value: ${self.starting_account_value:,.2f}")
        logger.info(f"Options Trading Level: {account_info.get('options_trading_level', 0)}")
        
        self.is_running = True
        
        try:
            while self.is_running:
                current_time = datetime.now(ZoneInfo("America/New_York"))
                
                if not self.is_market_hours():
                    logger.info("Market closed - waiting...")
                    time.sleep(300)  # Wait 5 minutes
                    continue
                
                # Update market data
                vix, spy_price = self.market_feed.update_market_data()
                
                # Monitor existing positions
                self.trader.monitor_positions()
                
                # Check for new trading opportunities
                if self.daily_trades < self.max_daily_trades:
                    is_high_conviction, conditions = self.analyzer.analyze_market_conditions(self.market_feed)
                    
                    if is_high_conviction:
                        logger.info(f"HIGH CONVICTION SETUP DETECTED - Score: {conditions['conviction_score']}")
                        
                        # Find optimal trade
                        setup = self.trader.find_optimal_diagonal_spread(spy_price, conditions['conviction_score'])
                        
                        if setup:
                            # Calculate position size
                            current_account_value = self.get_account_info().get('equity', 25000)
                            contracts = self.trader.calculate_position_size(
                                setup, current_account_value, conditions['conviction_score']
                            )
                            
                            # Execute trade
                            trade_result = self.trader.execute_diagonal_spread(setup, contracts)
                            
                            if trade_result:
                                self.daily_trades += 1
                                self.last_trade_time = current_time
                                
                                # Log trade details
                                logger.info(f"TRADE EXECUTED #{self.daily_trades}")
                                logger.info(f"Strategy: {trade_result['strategy']}")
                                logger.info(f"Contracts: {trade_result['contracts']}")
                                logger.info(f"Conviction Score: {trade_result['conviction_score']}")
                                logger.info(f"Short PUT: {trade_result['short_strike']}")
                                logger.info(f"Long PUT: {trade_result['long_strike']}")
                
                # Wait before next iteration
                time.sleep(60)  # Check every minute
                
                # Log hourly updates
                if current_time.minute == 0:
                    logger.info(f"Hourly Update - VIX: {vix:.2f}, SPY: ${spy_price:.2f}, "
                               f"Active Positions: {len(self.trader.active_positions)}")
        
        except KeyboardInterrupt:
            logger.info("Trading session interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error in trading session: {e}")
        finally:
            self.is_running = False
            self.log_daily_summary()
            logger.info("Trading session ended")

def main():
    """Main function to run the live trading bot."""
    
    # Verify environment variables
    if not os.getenv('ALPACA_API_KEY') or not os.getenv('ALPACA_SECRET_KEY'):
        print("Error: Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        print("You can get these from your Alpaca paper trading account at https://alpaca.markets/")
        return
    
    # Create and run trading bot
    bot = LiveTradingBot()
    bot.run_trading_session()

if __name__ == "__main__":
    main() 