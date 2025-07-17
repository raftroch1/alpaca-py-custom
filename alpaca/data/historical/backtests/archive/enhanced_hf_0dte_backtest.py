"""
ENHANCED HIGH FREQUENCY 0DTE BACKTEST
=====================================
Focus on WIN RATE and P&L performance on $25K account
Extended testing with detailed performance metrics

Key Features:
- Extended testing period (5-10 days)
- Detailed P&L tracking by day
- Win rate analysis
- Risk metrics (Sharpe, max drawdown)
- Realistic exit conditions
- Performance attribution
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from strategies.optimized_hf_0dte_strategy import OptimizedHighFrequency0DTEStrategy
import thetadata
from connector import ThetaClientConnector
import yfinance as yf

class EnhancedHighFrequencyBacktest:
    def __init__(self, starting_capital: float = 25000):
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        self.strategy = OptimizedHighFrequency0DTEStrategy()
        
        # Performance tracking
        self.trades = []
        self.daily_pnl = {}
        self.positions = {}
        self.position_counter = 0
        
        # Metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.max_drawdown = 0
        self.peak_capital = starting_capital
        
        # Enhanced logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'enhanced_hf_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )

    def download_extended_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download extended market data for comprehensive testing"""
        try:
            # Get trading days
            spy_data = yf.download('SPY', start=start_date, end=end_date, interval='1m')
            if spy_data.empty:
                logging.error("❌ No SPY data downloaded")
                return None
                
            # Get VIX data
            vix_data = yf.download('^VIX', start=start_date, end=end_date, interval='1d')
            
            # Merge and prepare data
            spy_data = spy_data.reset_index()
            spy_data['Date'] = spy_data['Datetime'].dt.date
            spy_data['Time'] = spy_data['Datetime'].dt.time
            
            # Add VIX levels
            vix_daily = {}
            for idx, row in vix_data.iterrows():
                vix_daily[idx.date()] = row['Close']
                
            spy_data['VIX'] = spy_data['Date'].map(vix_daily).fillna(16.0)
            
            # Filter trading hours (9:30 AM - 4:00 PM ET)
            spy_data = spy_data[
                (spy_data['Time'] >= pd.to_datetime('09:30:00').time()) &
                (spy_data['Time'] <= pd.to_datetime('16:00:00').time())
            ]
            
            logging.info(f"✅ Downloaded {len(spy_data)} minute bars for {len(spy_data['Date'].unique())} trading days")
            return spy_data
            
        except Exception as e:
            logging.error(f"❌ Data download error: {e}")
            return None

    def simulate_option_pricing(self, strike: float, option_type: str, spy_price: float, 
                               time_to_expiry: float, volatility: float = 0.20) -> float:
        """
        Enhanced option pricing simulation using Black-Scholes approximation
        """
        from scipy.stats import norm
        import math
        
        try:
            S = spy_price  # Current stock price
            K = strike     # Strike price
            T = time_to_expiry / 365.0  # Time to expiry in years
            r = 0.05       # Risk-free rate
            sigma = volatility  # Implied volatility
            
            if T <= 0:
                # At expiration
                if option_type == 'call':
                    return max(0, S - K)
                else:
                    return max(0, K - S)
            
            # Black-Scholes calculation
            d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
            d2 = d1 - sigma*math.sqrt(T)
            
            if option_type == 'call':
                price = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
            else:
                price = K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
                
            return max(0.01, price)  # Minimum price
            
        except:
            # Fallback intrinsic value
            if option_type == 'call':
                return max(0.05, spy_price - strike)
            else:
                return max(0.05, strike - spy_price)

    def execute_trade(self, signal: str, spy_price: float, vix_level: float, 
                     timestamp: str, analysis: dict) -> bool:
        """Enhanced trade execution with realistic option pricing"""
        
        # Find best option (simulated)
        if signal == 'BUY_CALL':
            strike = round(spy_price + 1.0, 0)  # Slightly OTM call
            option_type = 'call'
        else:
            strike = round(spy_price - 1.0, 0)  # Slightly OTM put
            option_type = 'put'
            
        # Calculate option price
        hours_to_expiry = 6.5  # Assume 6.5 hours to market close
        current_time = pd.to_datetime(timestamp).time()
        if current_time >= pd.to_datetime('15:30:00').time():
            hours_to_expiry = 1.0  # Last 30 minutes
        elif current_time >= pd.to_datetime('14:00:00').time():
            hours_to_expiry = 2.5
        elif current_time >= pd.to_datetime('12:00:00').time():
            hours_to_expiry = 4.0
            
        volatility = max(0.15, vix_level / 100)  # Convert VIX to volatility
        option_price = self.simulate_option_pricing(
            strike, option_type, spy_price, hours_to_expiry, volatility
        )
        
        # Check if option meets criteria
        if option_price < self.strategy.min_option_price or option_price > self.strategy.max_option_price:
            logging.warning(f"❌ Option price ${option_price:.2f} outside range ${self.strategy.min_option_price}-${self.strategy.max_option_price}")
            return False
            
        # Position sizing
        contracts = self.strategy.calculate_position_size(
            option_price, analysis['confidence'], self.current_capital
        )
        
        premium_paid = contracts * option_price * 100
        
        # Check available capital
        if premium_paid > self.current_capital * 0.20:  # Max 20% of capital per trade
            logging.warning(f"⚠️ Trade size too large: ${premium_paid:.2f}")
            return False
            
        # Execute trade
        self.position_counter += 1
        position_id = f"POS_{self.position_counter:04d}"
        
        position = {
            'id': position_id,
            'signal': signal,
            'entry_time': timestamp,
            'entry_price': option_price,
            'strike': strike,
            'option_type': option_type,
            'contracts': contracts,
            'premium_paid': premium_paid,
            'spy_price_entry': spy_price,
            'vix_entry': vix_level,
            'confidence': analysis['confidence'],
            'hours_to_expiry': hours_to_expiry,
            'status': 'OPEN'
        }
        
        self.positions[position_id] = position
        self.current_capital -= premium_paid
        self.total_trades += 1
        self.strategy.today_trades += 1
        
        logging.info(f"🎯 Executing Enhanced Trade #{self.total_trades}")
        logging.info(f"   💰 {signal} {contracts} contracts {strike}{option_type[0].upper()} @ ${option_price:.2f}")
        logging.info(f"   📊 Premium: ${premium_paid:.2f} | Confidence: {analysis['confidence']:.2f}")
        logging.info(f"   🕐 Time to expiry: {hours_to_expiry:.1f}h | Capital: ${self.current_capital:,.2f}")
        
        return True

    def check_exit_conditions(self, day_data: pd.DataFrame, current_time: str):
        """Enhanced exit condition checking with realistic option pricing"""
        
        current_row = day_data[day_data['Datetime'] == current_time].iloc[0]
        spy_price = current_row['Close']
        vix_level = current_row['VIX']
        
        positions_to_close = []
        
        for pos_id, position in self.positions.items():
            if position['status'] != 'OPEN':
                continue
                
            # Calculate time held and remaining
            entry_time = pd.to_datetime(position['entry_time'])
            current_dt = pd.to_datetime(current_time)
            time_held = (current_dt - entry_time).total_seconds() / 3600  # Hours
            
            # Calculate current option price
            remaining_hours = position['hours_to_expiry'] - time_held
            if remaining_hours <= 0:
                remaining_hours = 0.01
                
            volatility = max(0.15, vix_level / 100)
            current_option_price = self.simulate_option_pricing(
                position['strike'], position['option_type'], 
                spy_price, remaining_hours, volatility
            )
            
            # Check exit conditions
            should_exit, exit_reason, pnl_pct = self.strategy.should_exit_position(
                position['entry_price'], current_option_price, time_held, remaining_hours
            )
            
            if should_exit or remaining_hours <= 0.1:  # Force exit near expiration
                positions_to_close.append((pos_id, current_option_price, exit_reason))
                
        # Close positions
        for pos_id, exit_price, exit_reason in positions_to_close:
            self.close_position(pos_id, exit_price, exit_reason, current_time, spy_price)

    def close_position(self, position_id: str, exit_price: float, exit_reason: str, 
                      exit_time: str, spy_price: float):
        """Enhanced position closing with detailed P&L tracking"""
        
        position = self.positions[position_id]
        
        # Calculate P&L
        premium_received = position['contracts'] * exit_price * 100
        pnl = premium_received - position['premium_paid']
        pnl_pct = pnl / position['premium_paid']
        
        # Update capital
        self.current_capital += premium_received
        self.total_pnl += pnl
        
        # Track wins/losses
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
            
        # Update drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        else:
            current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Record trade
        trade_record = {
            'position_id': position_id,
            'signal': position['signal'],
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'strike': position['strike'],
            'option_type': position['option_type'],
            'contracts': position['contracts'],
            'premium_paid': position['premium_paid'],
            'premium_received': premium_received,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'confidence': position['confidence'],
            'spy_entry': position['spy_price_entry'],
            'spy_exit': spy_price,
            'capital_after': self.current_capital
        }
        
        self.trades.append(trade_record)
        position['status'] = 'CLOSED'
        
        # Update strategy performance tracking
        self.strategy.update_performance_metrics(pnl, exit_reason)
        
        logging.info(f"🔐 Closed {position_id}: {exit_reason}, P&L: ${pnl:+.2f} ({pnl_pct:+.1%})")

    def run_enhanced_backtest(self, start_date: str, end_date: str):
        """Run comprehensive backtest with extended data"""
        
        logging.info("🚀 ENHANCED HIGH FREQUENCY 0DTE BACKTEST")
        logging.info("="*60)
        
        # Initialize strategy
        if not self.strategy.initialize():
            logging.error("❌ Strategy initialization failed")
            return
            
        logging.info(f"📅 Period: {start_date} to {end_date}")
        logging.info(f"💰 Starting Capital: ${self.starting_capital:,.2f}")
        logging.info(f"🎯 Target: 8+ trades per day with positive win rate")
        logging.info(f"🔧 Optimizations: Expanded options, dynamic exits, better sizing")
        
        # Download data
        data = self.download_extended_data(start_date, end_date)
        if data is None:
            return
            
        # Group by trading days
        trading_days = data.groupby('Date')
        total_days = len(trading_days)
        
        # Process each day
        for day_num, (date, day_data) in enumerate(trading_days, 1):
            logging.info(f"\n📅 Processing Day {day_num}/{total_days}: {date}")
            
            day_start_capital = self.current_capital
            day_trades = 0
            day_signals = 0
            
            # Process each 5-minute interval
            intervals = range(0, len(day_data), 5)  # Every 5 minutes
            
            for i in intervals:
                if i >= len(day_data):
                    break
                    
                row = day_data.iloc[i]
                spy_price = row['Close']
                vix_level = row['VIX']
                timestamp = row['Datetime'].strftime('%Y-%m-%d %H:%M:%S')
                
                # Skip early market (before 10:00 AM)
                if row['Time'] < pd.to_datetime('10:00:00').time():
                    continue
                    
                # Skip late market (after 3:30 PM)  
                if row['Time'] > pd.to_datetime('15:30:00').time():
                    continue
                
                # Check exit conditions for open positions
                self.check_exit_conditions(day_data, timestamp)
                
                # Analyze market conditions
                current_data = day_data.iloc[:i+1]
                analysis = self.strategy.analyze_market_conditions(
                    spy_price, vix_level, timestamp, current_data
                )
                
                if analysis['signal'] != 'HOLD':
                    day_signals += 1
                    
                    logging.info(f"🎯 Enhanced Signal: {analysis['signal']}, Confidence: {analysis['confidence']:.2f}")
                    logging.info(f"   📊 Factors - Bullish: {analysis['bullish_factors']:.1f}, Bearish: {analysis['bearish_factors']:.1f}")
                    logging.info(f"   📈 RSI: {analysis['rsi']:.1f}, VIX: {analysis['vix']:.1f}, Volume Surge: {analysis['volume_surge']}")
                    
                    # Execute trade
                    if self.execute_trade(analysis['signal'], spy_price, vix_level, timestamp, analysis):
                        day_trades += 1
                        
            # Close all positions at end of day
            final_row = day_data.iloc[-1]
            self.check_exit_conditions(day_data, final_row['Datetime'].strftime('%Y-%m-%d %H:%M:%S'))
            
            # Force close any remaining positions
            for pos_id, position in self.positions.items():
                if position['status'] == 'OPEN':
                    exit_price = 0.01  # Expired worthless
                    self.close_position(pos_id, exit_price, "EXPIRATION", 
                                      final_row['Datetime'].strftime('%Y-%m-%d %H:%M:%S'), 
                                      final_row['Close'])
            
            # Day summary
            day_pnl = self.current_capital - day_start_capital
            self.daily_pnl[str(date)] = day_pnl
            
            logging.info(f"📊 Day Summary: {day_trades} trades, ${day_pnl:+.2f} P&L, {day_signals} signals")
            
        # Final results
        self.print_enhanced_results()
        self.save_detailed_results()

    def print_enhanced_results(self):
        """Print comprehensive backtest results"""
        
        total_return = (self.current_capital - self.starting_capital) / self.starting_capital
        trading_days = len(self.daily_pnl)
        
        logging.info("\n" + "="*80)
        logging.info("🎯 ENHANCED HIGH FREQUENCY 0DTE BACKTEST RESULTS")
        logging.info("="*80)
        
        # Capital metrics
        logging.info(f"💰 Starting Capital: ${self.starting_capital:,.2f}")
        logging.info(f"💰 Final Capital: ${self.current_capital:,.2f}")
        logging.info(f"📊 Total Return: {total_return:.2%}")
        logging.info(f"💵 Total P&L: ${self.total_pnl:+,.2f}")
        
        # Trading metrics
        logging.info(f"📅 Trading Days: {trading_days}")
        logging.info(f"🎯 Total Trades: {self.total_trades}")
        trades_per_day = self.total_trades / trading_days if trading_days > 0 else 0
        logging.info(f"⚡ Trades per Day: {trades_per_day:.1f}")
        
        if trades_per_day >= 8.0:
            logging.info("✅ Frequency target ACHIEVED!")
        else:
            logging.info(f"⚠️ Frequency below target ({trades_per_day:.1f} vs 8.0)")
            
        # Win rate analysis
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        logging.info(f"\n📊 WIN RATE ANALYSIS:")
        logging.info(f"   ✅ Win Rate: {win_rate:.1%} ({self.winning_trades}/{self.total_trades})")
        
        if win_rate >= 0.4:
            logging.info("✅ Win rate target ACHIEVED!")
        else:
            logging.info(f"⚠️ Win rate needs improvement (target: 40%+)")
            
        # P&L analysis
        if len(self.trades) > 0:
            df_trades = pd.DataFrame(self.trades)
            
            winning_trades = df_trades[df_trades['pnl'] > 0]
            losing_trades = df_trades[df_trades['pnl'] < 0]
            
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            
            logging.info(f"   💚 Average Win: ${avg_win:.2f}")
            logging.info(f"   💔 Average Loss: ${avg_loss:.2f}")
            
            if avg_loss != 0:
                profit_factor = abs(avg_win * len(winning_trades)) / abs(avg_loss * len(losing_trades))
                logging.info(f"   📈 Profit Factor: {profit_factor:.2f}")
                
        # Risk metrics
        logging.info(f"\n🛡️ RISK METRICS:")
        logging.info(f"   📉 Max Drawdown: {self.max_drawdown:.2%}")
        
        if len(self.daily_pnl) > 1:
            daily_returns = np.array(list(self.daily_pnl.values())) / self.starting_capital
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            logging.info(f"   📊 Sharpe Ratio: {sharpe_ratio:.2f}")
            
        # Exit analysis
        if len(self.trades) > 0:
            df_trades = pd.DataFrame(self.trades)
            exit_analysis = df_trades.groupby('exit_reason').agg({
                'pnl': ['count', 'sum', 'mean']
            }).round(2)
            
            logging.info(f"\n🚪 EXIT ANALYSIS:")
            for exit_reason in exit_analysis.index:
                count = exit_analysis.loc[exit_reason, ('pnl', 'count')]
                total_pnl = exit_analysis.loc[exit_reason, ('pnl', 'sum')]
                pct = count / len(df_trades) * 100
                logging.info(f"   {exit_reason}: {count} trades ({pct:.1f}%), ${total_pnl:+.2f} P&L")

    def save_detailed_results(self):
        """Save detailed results to CSV files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save trades
        if len(self.trades) > 0:
            df_trades = pd.DataFrame(self.trades)
            filename = f"enhanced_hf_0dte_trades_{timestamp}.csv"
            df_trades.to_csv(filename, index=False)
            logging.info(f"💾 Trade details saved to: {filename}")
            
        # Save daily summary
        if len(self.daily_pnl) > 0:
            daily_summary = []
            cumulative_pnl = 0
            
            for date, pnl in self.daily_pnl.items():
                cumulative_pnl += pnl
                daily_summary.append({
                    'date': date,
                    'daily_pnl': pnl,
                    'cumulative_pnl': cumulative_pnl,
                    'capital': self.starting_capital + cumulative_pnl
                })
                
            df_daily = pd.DataFrame(daily_summary)
            filename = f"enhanced_hf_0dte_daily_{timestamp}.csv"
            df_daily.to_csv(filename, index=False)
            logging.info(f"💾 Daily summary saved to: {filename}")
            
        logging.info("="*80)

if __name__ == "__main__":
    # Run enhanced backtest
    backtest = EnhancedHighFrequencyBacktest(starting_capital=25000)
    
    # Extended testing period (last 5 trading days)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    
    backtest.run_enhanced_backtest(start_date, end_date) 