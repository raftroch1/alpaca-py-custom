#!/usr/bin/env python3
"""
HIGH FREQUENCY 0DTE STRATEGY BACKTEST

Comprehensive backtesting framework for the high-frequency 0DTE options strategy.
Designed to validate the 8+ trades per day target with detailed performance analytics.

Features:
- Minute-by-minute SPY data analysis
- Real ThetaData option pricing
- Intraday position management
- Stop loss and profit target simulation
- Comprehensive performance reporting
- Daily and trade-level analytics
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
import warnings
import logging

# Add paths for imports
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(os.path.dirname(current_dir), 'strategies'))
sys.path.append(os.path.join(os.path.dirname(current_dir), 'thetadata'))

from high_frequency_0dte_strategy import HighFrequency0DTEStrategy
from connector import ThetaDataConnector

warnings.filterwarnings('ignore')

class HighFrequency0DTEBacktest:
    """
    High Frequency 0DTE Options Strategy Backtest
    
    Simulates intraday trading with multiple trades per day, including
    stop losses, profit targets, and end-of-day position management.
    """
    
    def __init__(self, starting_capital: float = 25000):
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        
        # Initialize strategy
        self.strategy = HighFrequency0DTEStrategy(starting_capital=starting_capital)
        self.theta_connector = ThetaDataConnector()
        
        # Backtest tracking
        self.trades = []
        self.daily_summary = {}
        self.open_positions = {}
        self.position_id_counter = 0
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup comprehensive logging for backtest."""
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'high_frequency_0dte_backtest_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_backtest(self, start_date: str = '2025-07-10', end_date: str = '2025-07-12'):
        """
        Run comprehensive high-frequency backtest.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        self.logger.info("🚀 STARTING HIGH FREQUENCY 0DTE BACKTEST")
        self.logger.info("=" * 70)
        self.logger.info(f"📅 Period: {start_date} to {end_date}")
        self.logger.info(f"💰 Starting Capital: ${self.starting_capital:,.2f}")
        self.logger.info("🎯 Target: 8+ trades per day")
        
        # Get market data
        market_data = self.get_market_data(start_date, end_date)
        if market_data.empty:
            self.logger.error("❌ No market data available")
            return
        
        # Process each trading day
        trading_days = market_data.groupby(market_data.index.date)
        total_days = len(trading_days)
        
        for day_count, (trade_date, day_data) in enumerate(trading_days, 1):
            self.logger.info(f"\n📅 Processing Day {day_count}/{total_days}: {trade_date}")
            
            # Reset daily counters
            self.strategy.daily_trade_count = 0
            self.strategy.last_trade_date = trade_date
            
            # Process intraday trading
            daily_results = self.process_trading_day(day_data, trade_date)
            self.daily_summary[str(trade_date)] = daily_results
            
            # Log daily progress
            if day_count % 5 == 0:
                self.print_progress_update(day_count, total_days)
        
        # Generate final results
        self.generate_comprehensive_results()
    
    def get_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get minute-by-minute SPY data for analysis."""
        try:
            import yfinance as yf
            
            # Download SPY minute data
            spy_ticker = yf.Ticker("SPY")
            spy_data = spy_ticker.history(
                start=start_date,
                end=end_date,
                interval="1m",
                auto_adjust=True,
                prepost=False
            )
            
            if spy_data.empty:
                self.logger.error("❌ No SPY data downloaded")
                return pd.DataFrame()
            
            # Download VIX data (daily)
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(start=start_date, end=end_date)
            
            # Merge VIX with SPY data (forward fill VIX for minute data)
            spy_data['vix'] = np.nan
            for date in vix_data.index.date:
                spy_data.loc[spy_data.index.date == date, 'vix'] = vix_data.loc[vix_data.index.date == date, 'Close'].iloc[0]
            
            spy_data['vix'] = spy_data['vix'].fillna(method='ffill')
            
            # Filter to market hours (9:30 AM - 4:00 PM ET)
            spy_data = spy_data.between_time('09:30', '16:00')
            
            unique_dates = pd.Series(spy_data.index.date).nunique()
            self.logger.info(f"✅ Downloaded {len(spy_data)} minute bars for {unique_dates} trading days")
            
            return spy_data
            
        except Exception as e:
            self.logger.error(f"❌ Error downloading market data: {e}")
            return pd.DataFrame()
    
    def process_trading_day(self, day_data: pd.DataFrame, trade_date) -> Dict:
        """
        Process a single trading day with intraday analysis.
        
        Args:
            day_data: Minute-by-minute data for the day
            trade_date: Trading date
            
        Returns:
            Dictionary with daily summary
        """
        day_start_capital = self.current_capital
        daily_trades = 0
        daily_pnl = 0
        signals_generated = 0
        
        # Track positions opened today
        todays_positions = []
        
        # Process every 5 minutes during market hours
        for i in range(0, len(day_data), 5):  # Every 5 minutes
            current_time = day_data.index[i]
            current_bar = day_data.iloc[i]
            
            spy_price = float(current_bar['Close'])
            vix_level = float(current_bar['vix'])
            date_str = trade_date.strftime('%Y-%m-%d')
            
            # Check existing positions for exits
            self.check_position_exits(current_time, spy_price)
            
            # Skip new signals after 2:00 PM to allow for position management
            if current_time.time() > time(14, 0):
                continue
            
            # Analyze market conditions with pre-loaded market data
            try:
                # Pass the market data up to current time to avoid lookahead bias
                current_data = day_data.iloc[:i+1]  # Only data up to current time
                analysis = self.strategy.analyze_market_conditions(spy_price, vix_level, date_str, current_data)
                
                if analysis['signal'] != 'HOLD':
                    signals_generated += 1
                    
                    # Execute strategy
                    trade_result = self.strategy.execute_strategy(analysis, spy_price, date_str)
                    
                    if trade_result:
                        # Create position
                        position = self.create_position(trade_result, current_time, spy_price)
                        todays_positions.append(position['id'])
                        daily_trades += 1
                        
                        self.logger.info(f"⭐ Trade #{daily_trades}: {trade_result['signal']} at {current_time.strftime('%H:%M')}")
                        
            except Exception as e:
                self.logger.warning(f"⚠️  Error in analysis at {current_time}: {e}")
        
        # Close all remaining positions at end of day (0DTE expiration)
        daily_pnl += self.close_eod_positions(trade_date, day_data.iloc[-1]['Close'])
        
        # Update capital
        self.current_capital = day_start_capital + daily_pnl
        
        daily_summary = {
            'date': str(trade_date),
            'starting_capital': day_start_capital,
            'ending_capital': self.current_capital,
            'daily_pnl': daily_pnl,
            'daily_trades': daily_trades,
            'signals_generated': signals_generated,
            'avg_spy_price': day_data['Close'].mean(),
            'avg_vix': day_data['vix'].mean(),
            'positions_opened': len(todays_positions),
            'signal_conversion_rate': daily_trades / signals_generated if signals_generated > 0 else 0
        }
        
        self.logger.info(f"📊 Day Summary: {daily_trades} trades, ${daily_pnl:.2f} P&L, {signals_generated} signals")
        
        return daily_summary
    
    def create_position(self, trade_result: Dict, entry_time: datetime, current_price: float) -> Dict:
        """Create a new position from trade result."""
        self.position_id_counter += 1
        position_id = f"POS_{self.position_id_counter:04d}"
        
        position = {
            'id': position_id,
            'signal': trade_result['signal'],
            'entry_time': entry_time,
            'entry_price': trade_result['option_price'],
            'contracts': trade_result['contracts'],
            'premium_paid': trade_result['premium'],
            'strike': trade_result['strike'],
            'right': trade_result['right'],
            'spy_entry_price': current_price,
            'stop_loss_target': trade_result['stop_loss_target'],
            'profit_target': trade_result['profit_target'],
            'confidence': trade_result['confidence'],
            'status': 'OPEN'
        }
        
        self.open_positions[position_id] = position
        return position
    
    def check_position_exits(self, current_time: datetime, spy_price: float):
        """Check all open positions for exit conditions."""
        positions_to_close = []
        
        for pos_id, position in self.open_positions.items():
            if position['status'] != 'OPEN':
                continue
            
            # Simulate current option value (simplified)
            current_option_value = self.estimate_option_value(position, spy_price)
            current_position_value = current_option_value * position['contracts'] * 100
            
            exit_reason = None
            
            # Check stop loss
            if current_position_value <= position['stop_loss_target']:
                exit_reason = 'STOP_LOSS'
            
            # Check profit target
            elif current_position_value >= position['profit_target']:
                exit_reason = 'PROFIT_TARGET'
            
            # Check time-based exit (after 2:30 PM)
            elif current_time.time() > time(14, 30):
                exit_reason = 'TIME_EXIT'
            
            if exit_reason:
                positions_to_close.append((pos_id, current_option_value, exit_reason, current_time))
        
        # Close positions
        for pos_id, exit_value, exit_reason, exit_time in positions_to_close:
            self.close_position(pos_id, exit_value, exit_reason, exit_time)
    
    def estimate_option_value(self, position: Dict, spy_price: float) -> float:
        """Estimate current option value (simplified model)."""
        strike = position['strike']
        right = position['right']
        entry_price = position['entry_price']
        
        # Simple intrinsic value calculation
        if right == 'C':
            intrinsic = max(0, spy_price - strike)
        else:
            intrinsic = max(0, strike - spy_price)
        
        # Add time value (decreasing throughout day)
        time_value = entry_price * 0.3  # Simplified time decay
        
        return max(0.01, intrinsic + time_value)  # Minimum 1 cent
    
    def close_position(self, position_id: str, exit_value: float, exit_reason: str, exit_time: datetime):
        """Close a position and record the trade."""
        position = self.open_positions[position_id]
        
        exit_premium = exit_value * position['contracts'] * 100
        pnl = exit_premium - position['premium_paid']
        
        # Record completed trade
        trade_record = {
            'position_id': position_id,
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'signal': position['signal'],
            'strike': position['strike'],
            'right': position['right'],
            'contracts': position['contracts'],
            'entry_price': position['entry_price'],
            'exit_price': exit_value,
            'premium_paid': position['premium_paid'],
            'premium_received': exit_premium,
            'pnl': pnl,
            'return_pct': (pnl / position['premium_paid']) * 100,
            'exit_reason': exit_reason,
            'confidence': position['confidence'],
            'holding_time_minutes': (exit_time - position['entry_time']).total_seconds() / 60
        }
        
        self.trades.append(trade_record)
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        
        # Mark position as closed
        position['status'] = 'CLOSED'
        position['exit_reason'] = exit_reason
        position['pnl'] = pnl
        
        self.logger.info(f"🔐 Closed {position_id}: {exit_reason}, P&L: ${pnl:.2f}")
    
    def close_eod_positions(self, trade_date, final_spy_price: float) -> float:
        """Close all remaining positions at end of day (0DTE expiration)."""
        eod_pnl = 0
        
        for pos_id, position in self.open_positions.items():
            if position['status'] == 'OPEN':
                # 0DTE options expire worthless if not ITM
                strike = position['strike']
                right = position['right']
                
                if right == 'C':
                    final_value = max(0, final_spy_price - strike)
                else:
                    final_value = max(0, strike - final_spy_price)
                
                self.close_position(pos_id, final_value, 'EOD_EXPIRATION', 
                                  datetime.combine(trade_date, time(16, 0)))
        
        return eod_pnl
    
    def print_progress_update(self, current_day: int, total_days: int):
        """Print progress update during backtest."""
        progress_pct = (current_day / total_days) * 100
        
        self.logger.info(f"\n📈 PROGRESS UPDATE ({progress_pct:.1f}%)")
        self.logger.info(f"   Days Processed: {current_day}/{total_days}")
        self.logger.info(f"   Total Trades: {self.total_trades}")
        self.logger.info(f"   Current Capital: ${self.current_capital:,.2f}")
        self.logger.info(f"   Total P&L: ${self.total_pnl:,.2f}")
        
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
            self.logger.info(f"   Win Rate: {win_rate:.1f}%")
    
    def generate_comprehensive_results(self):
        """Generate comprehensive backtest results and analytics."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🎯 HIGH FREQUENCY 0DTE BACKTEST RESULTS")
        self.logger.info("=" * 80)
        
        # Overall performance
        total_return = ((self.current_capital - self.starting_capital) / self.starting_capital) * 100
        trading_days = len(self.daily_summary)
        
        self.logger.info(f"💰 Starting Capital: ${self.starting_capital:,.2f}")
        self.logger.info(f"💰 Final Capital: ${self.current_capital:,.2f}")
        self.logger.info(f"📊 Total Return: {total_return:.2f}%")
        self.logger.info(f"💵 Total P&L: ${self.total_pnl:,.2f}")
        self.logger.info(f"📅 Trading Days: {trading_days}")
        self.logger.info(f"🎯 Total Trades: {self.total_trades}")
        
        if trading_days > 0:
            trades_per_day = self.total_trades / trading_days
            self.logger.info(f"⚡ Trades per Day: {trades_per_day:.1f}")
            
            # Check if we achieved our frequency target
            if trades_per_day >= 8:
                self.logger.info("✅ FREQUENCY TARGET ACHIEVED (8+ trades/day)")
            else:
                self.logger.info(f"⚠️  Frequency below target ({trades_per_day:.1f} vs 8.0)")
        
        # Trade analysis
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
            losing_trades = self.total_trades - self.winning_trades
            
            trades_df = pd.DataFrame(self.trades)
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if self.winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            
            self.logger.info(f"\n📊 TRADE ANALYSIS:")
            self.logger.info(f"   ✅ Win Rate: {win_rate:.1f}% ({self.winning_trades}/{self.total_trades})")
            self.logger.info(f"   💚 Average Win: ${avg_win:.2f}")
            self.logger.info(f"   💔 Average Loss: ${avg_loss:.2f}")
            
            if avg_loss != 0:
                profit_factor = abs(avg_win * self.winning_trades) / abs(avg_loss * losing_trades)
                self.logger.info(f"   📈 Profit Factor: {profit_factor:.2f}")
            
            # Exit reason analysis
            exit_reasons = trades_df['exit_reason'].value_counts()
            self.logger.info(f"\n🚪 EXIT ANALYSIS:")
            for reason, count in exit_reasons.items():
                pct = (count / len(trades_df)) * 100
                reason_pnl = trades_df[trades_df['exit_reason'] == reason]['pnl'].sum()
                self.logger.info(f"   {reason}: {count} trades ({pct:.1f}%), ${reason_pnl:.2f} P&L")
            
            # Signal analysis
            signals = trades_df['signal'].value_counts()
            self.logger.info(f"\n📊 SIGNAL ANALYSIS:")
            for signal, count in signals.items():
                signal_pnl = trades_df[trades_df['signal'] == signal]['pnl'].sum()
                signal_win_rate = len(trades_df[(trades_df['signal'] == signal) & (trades_df['pnl'] > 0)]) / count * 100
                self.logger.info(f"   {signal}: {count} trades, ${signal_pnl:.2f} P&L, {signal_win_rate:.1f}% win rate")
        
        # Save results to CSV
        self.save_results_to_csv()
        
        self.logger.info("=" * 80)
    
    def save_results_to_csv(self):
        """Save detailed results to CSV files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save trade details
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = f'high_frequency_0dte_trades_{timestamp}.csv'
            trades_df.to_csv(trades_file, index=False)
            self.logger.info(f"💾 Trade details saved to: {trades_file}")
        
        # Save daily summary
        if self.daily_summary:
            daily_df = pd.DataFrame(list(self.daily_summary.values()))
            daily_file = f'high_frequency_0dte_daily_summary_{timestamp}.csv'
            daily_df.to_csv(daily_file, index=False)
            self.logger.info(f"💾 Daily summary saved to: {daily_file}")

# Run backtest if executed directly
if __name__ == "__main__":
    print("🚀 HIGH FREQUENCY 0DTE STRATEGY BACKTEST")
    print("=" * 60)
    
    backtest = HighFrequency0DTEBacktest(starting_capital=25000)
    
    # Run backtest for a 2-day period (using recent available data)
    backtest.run_backtest(
        start_date='2025-07-10',
        end_date='2025-07-12'
    ) 