"""
OPTIMIZED COMBINED HIGH FREQUENCY 0DTE BACKTEST
===============================================
Combines Option A (Optimization) + Option B (Extended Testing)
Focus on WIN RATE and P&L performance on $25K account

Key Optimizations:
✅ Expanded option range: $0.30-$5.00 (vs $0.50-$3.00)
✅ Lower confidence threshold: 0.35 (vs 0.4) - 12.5% more signals
✅ Enhanced exit conditions: Multiple profit targets + dynamic stops
✅ Better position sizing: Risk-adjusted based on confidence
✅ Extended testing: 5+ trading days vs 2 days
✅ Detailed P&L tracking: Daily, per-trade, win rate analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import yfinance as yf
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class OptimizedHighFrequency0DTEStrategy:
    """Optimized version focusing on WIN RATE and P&L"""
    
    def __init__(self):
        # OPTIMIZED PARAMETERS
        self.confidence_threshold = 0.35    # Lowered from 0.4 (12.5% more sensitive)
        self.factor_threshold = 1.3         # Lowered from 1.5 (13% more sensitive)
        
        # EXPANDED OPTION SELECTION
        self.min_option_price = 0.30        # Expanded from 0.50
        self.max_option_price = 5.00        # Expanded from 3.00
        
        # ENHANCED RISK MANAGEMENT
        self.max_risk_per_trade = 0.015     # 1.5% of account per trade
        self.max_daily_trades = 20          # Increased from 15
        self.max_position_size = 15         # Increased from 10
        
        # IMPROVED EXIT CONDITIONS
        self.profit_target_quick = 0.30     # 30% quick profit
        self.profit_target_main = 0.60      # 60% main target
        self.profit_target_home = 1.20      # 120% home run
        self.stop_loss = 0.35               # 35% stop (tighter)
        
        # Enhanced technical parameters
        self.rsi_period = 12
        self.rsi_oversold = 25
        self.rsi_overbought = 75
        self.ema_fast = 8
        self.ema_slow = 18
        self.volume_surge_threshold = 1.2
        
        # Performance tracking
        self.today_trades = 0
        self.last_trade_date = None

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def analyze_market_conditions(self, spy_price: float, vix_level: float, 
                                date: str, market_data: pd.DataFrame = None) -> Dict[str, Any]:
        """OPTIMIZED market analysis with enhanced signal generation"""
        
        # Reset daily trade count
        current_date = date.split()[0]
        if self.last_trade_date != current_date:
            self.today_trades = 0
            self.last_trade_date = current_date
            
        if self.today_trades >= self.max_daily_trades:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'daily_limit_reached'}
            
        # Use pre-loaded market data
        if market_data is not None and len(market_data) > 0:
            df = market_data.tail(25).copy()  # Use last 25 periods
        else:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'no_data'}
        
        if len(df) < 8:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'insufficient_data'}
            
        # Normalize column names
        if 'Close' in df.columns:
            df = df.rename(columns={'Close': 'close', 'Volume': 'volume', 'High': 'high', 'Low': 'low'})
        
        # Enhanced technical analysis
        current_price = df['close'].iloc[-1]
        
        # RSI calculation (more sensitive)
        rsi = self.calculate_rsi(df['close'], self.rsi_period)
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        # EMA signals (faster response)
        ema_fast = df['close'].ewm(span=self.ema_fast).mean()
        ema_slow = df['close'].ewm(span=self.ema_slow).mean()
        ema_bullish = ema_fast.iloc[-1] > ema_slow.iloc[-1]
        ema_momentum = (ema_fast.iloc[-1] / ema_slow.iloc[-1] - 1) * 100
        
        # Volume analysis (enhanced)
        if 'volume' in df.columns:
            recent_volume = df['volume'].iloc[-2:].mean()
            avg_volume = df['volume'].iloc[-8:].mean()
            volume_surge = recent_volume > avg_volume * self.volume_surge_threshold
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        else:
            volume_surge = False
            volume_ratio = 1
        
        # Price momentum (multiple timeframes)
        momentum_1 = (current_price / df['close'].iloc[-2] - 1) * 100 if len(df) >= 2 else 0
        momentum_3 = (current_price / df['close'].iloc[-4] - 1) * 100 if len(df) >= 4 else momentum_1
        
        # ENHANCED SIGNAL CALCULATION
        bullish_factors = 0
        bearish_factors = 0
        
        # RSI factors (more granular)
        if current_rsi <= self.rsi_oversold:
            bullish_factors += 2.5
        elif current_rsi <= 30:
            bullish_factors += 1.5
        elif current_rsi <= 35:
            bullish_factors += 1.0
        elif current_rsi >= self.rsi_overbought:
            bearish_factors += 2.5
        elif current_rsi >= 70:
            bearish_factors += 1.5
        elif current_rsi >= 65:
            bearish_factors += 1.0
            
        # EMA factors (enhanced)
        if ema_bullish and ema_momentum > 0.3:
            bullish_factors += 2.0
        elif ema_bullish and ema_momentum > 0.1:
            bullish_factors += 1.2
        elif ema_bullish:
            bullish_factors += 0.8
        elif not ema_bullish and ema_momentum < -0.3:
            bearish_factors += 2.0
        elif not ema_bullish and ema_momentum < -0.1:
            bearish_factors += 1.2
        else:
            bearish_factors += 0.8
            
        # Volume factors (increased weight)
        if volume_surge:
            if momentum_1 > 0.2:
                bullish_factors += 1.5
            elif momentum_1 < -0.2:
                bearish_factors += 1.5
            else:
                bullish_factors += 0.7
                bearish_factors += 0.7
                
        # VIX factors (enhanced)
        if vix_level > 25 and current_rsi < 40:
            bullish_factors += 2.0  # High fear + oversold
        elif vix_level > 20 and current_rsi < 35:
            bullish_factors += 1.2
        elif vix_level < 15 and current_rsi > 65:
            bearish_factors += 1.5  # Complacency + overbought
        elif vix_level > 20:
            bearish_factors += 0.5
            
        # Momentum factors
        if momentum_1 > 0.4 and momentum_3 > 0.6:
            bullish_factors += 1.5
        elif momentum_1 < -0.4 and momentum_3 < -0.6:
            bearish_factors += 1.5
            
        # Calculate signal strength
        net_bullish = bullish_factors - bearish_factors * 0.7
        net_bearish = bearish_factors - bullish_factors * 0.7
        
        # Generate signals with optimized threshold
        signal = 'HOLD'
        confidence = 0
        
        if net_bullish >= self.factor_threshold:
            signal = 'BUY_CALL'
            confidence = min(0.95, 0.4 + (net_bullish - self.factor_threshold) * 0.25)
        elif net_bearish >= self.factor_threshold:
            signal = 'BUY_PUT'
            confidence = min(0.95, 0.4 + (net_bearish - self.factor_threshold) * 0.25)
            
        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            signal = 'HOLD'
            confidence = 0
            
        return {
            'signal': signal,
            'confidence': confidence,
            'bullish_factors': bullish_factors,
            'bearish_factors': bearish_factors,
            'rsi': current_rsi,
            'vix': vix_level,
            'volume_surge': volume_surge,
            'volume_ratio': volume_ratio,
            'momentum_1': momentum_1,
            'momentum_3': momentum_3,
            'ema_momentum': ema_momentum,
            'net_bullish': net_bullish,
            'net_bearish': net_bearish
        }

    def calculate_position_size(self, option_price: float, confidence: float, 
                              account_value: float = 25000) -> int:
        """ENHANCED position sizing with better risk management"""
        
        # Base position calculation
        max_risk_amount = account_value * self.max_risk_per_trade
        base_contracts = max(1, int(max_risk_amount / (option_price * 100)))
        
        # Confidence adjustment (more aggressive scaling)
        confidence_multiplier = 0.7 + (confidence * 0.8)  # 0.7 to 1.5x
        adjusted_contracts = int(base_contracts * confidence_multiplier)
        
        # Option price adjustment
        if option_price < 0.75:
            price_multiplier = 1.4  # Favor very cheap options
        elif option_price < 1.50:
            price_multiplier = 1.2
        elif option_price < 3.00:
            price_multiplier = 1.0
        else:
            price_multiplier = 0.8  # Reduce size for expensive options
            
        final_contracts = int(adjusted_contracts * price_multiplier)
        
        # Apply limits
        final_contracts = min(final_contracts, self.max_position_size)
        final_contracts = max(final_contracts, 1)
        
        return final_contracts

class OptimizedCombinedBacktest:
    """Enhanced backtest combining optimization + extended testing"""
    
    def __init__(self, starting_capital: float = 25000):
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        self.strategy = OptimizedHighFrequency0DTEStrategy()
        
        # Performance tracking
        self.trades = []
        self.daily_pnl = {}
        self.positions = {}
        self.position_counter = 0
        
        # Enhanced metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        self.max_drawdown = 0
        self.peak_capital = starting_capital
        self.daily_returns = []

    def download_extended_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Download extended market data for comprehensive testing"""
        try:
            logging.info(f"📊 Downloading extended data from {start_date} to {end_date}...")
            
            # Download SPY minute data
            spy_data = yf.download('SPY', start=start_date, end=end_date, interval='1m')
            if spy_data.empty:
                logging.error("❌ No SPY data downloaded")
                return None
                
            # Download VIX daily data
            vix_data = yf.download('^VIX', start=start_date, end=end_date, interval='1d')
            
            # Prepare data
            spy_data = spy_data.reset_index()
            spy_data['Date'] = spy_data['Datetime'].dt.date
            spy_data['Time'] = spy_data['Datetime'].dt.time
            
            # Map VIX to each minute
            vix_daily = {}
            for idx, row in vix_data.iterrows():
                vix_daily[idx.date()] = row['Close']
                
            spy_data['VIX'] = spy_data['Date'].map(vix_daily).fillna(16.0)
            
            # Filter to market hours (9:30 AM - 4:00 PM ET)
            spy_data = spy_data[
                (spy_data['Time'] >= pd.to_datetime('09:30:00').time()) &
                (spy_data['Time'] <= pd.to_datetime('16:00:00').time())
            ]
            
            # Remove low volume periods
            spy_data = spy_data[spy_data['Volume'] > 0]
            
            trading_days = len(spy_data['Date'].unique())
            logging.info(f"✅ Downloaded {len(spy_data)} minute bars for {trading_days} trading days")
            return spy_data
            
        except Exception as e:
            logging.error(f"❌ Data download error: {e}")
            return None

    def simulate_realistic_option_pricing(self, strike: float, option_type: str, 
                                        spy_price: float, hours_to_expiry: float, 
                                        volatility: float = 0.20) -> float:
        """Realistic option pricing with time decay and volatility"""
        
        # Simple Black-Scholes approximation
        time_value_factor = max(0.1, hours_to_expiry / 6.5)  # Normalize to trading day
        volatility_factor = volatility / 100 if volatility > 1 else volatility
        
        # Intrinsic value
        if option_type == 'call':
            intrinsic = max(0, spy_price - strike)
        else:
            intrinsic = max(0, strike - spy_price)
            
        # Time value (decreases as expiration approaches)
        moneyness = abs(spy_price - strike) / spy_price
        time_value = (0.5 + volatility_factor * 2) * time_value_factor * (1 - moneyness * 2)
        time_value = max(0, time_value)
        
        # Total option value
        option_price = intrinsic + time_value
        
        # Minimum price
        option_price = max(0.05, option_price)
        
        # Add some randomness for realism
        noise = np.random.normal(0, 0.02)  # 2% noise
        option_price *= (1 + noise)
        
        return max(0.05, option_price)

    def execute_optimized_trade(self, signal: str, spy_price: float, vix_level: float, 
                              timestamp: str, analysis: Dict[str, Any]) -> bool:
        """Execute trade with optimized parameters"""
        
        # Determine strike (more aggressive positioning)
        if signal == 'BUY_CALL':
            # Slightly OTM calls
            strike = round(spy_price + np.random.uniform(0.5, 2.0), 0)
            option_type = 'call'
        else:
            # Slightly OTM puts
            strike = round(spy_price - np.random.uniform(0.5, 2.0), 0)
            option_type = 'put'
            
        # Calculate realistic option pricing
        current_time = pd.to_datetime(timestamp).time()
        market_open = pd.to_datetime('09:30:00').time()
        market_close = pd.to_datetime('16:00:00').time()
        
        # Calculate hours to expiry
        time_elapsed = (pd.to_datetime(current_time.strftime('%H:%M:%S')) - 
                       pd.to_datetime(market_open.strftime('%H:%M:%S'))).total_seconds() / 3600
        hours_to_expiry = 6.5 - time_elapsed  # 6.5 hour trading day
        hours_to_expiry = max(0.1, hours_to_expiry)
        
        volatility = max(15, vix_level) / 100  # Convert VIX to volatility
        option_price = self.simulate_realistic_option_pricing(
            strike, option_type, spy_price, hours_to_expiry, volatility
        )
        
        # Check expanded price range
        if option_price < self.strategy.min_option_price or option_price > self.strategy.max_option_price:
            logging.warning(f"❌ Option price ${option_price:.2f} outside expanded range ${self.strategy.min_option_price}-${self.strategy.max_option_price}")
            return False
            
        # Enhanced position sizing
        contracts = self.strategy.calculate_position_size(
            option_price, analysis['confidence'], self.current_capital
        )
        
        premium_paid = contracts * option_price * 100
        
        # Check capital availability (more conservative)
        if premium_paid > self.current_capital * 0.25:  # Max 25% per trade
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
        
        logging.info(f"🎯 OPTIMIZED Trade #{self.total_trades}")
        logging.info(f"   💰 {signal} {contracts} contracts {strike}{option_type[0].upper()} @ ${option_price:.2f}")
        logging.info(f"   📊 Premium: ${premium_paid:.2f} | Confidence: {analysis['confidence']:.2f}")
        logging.info(f"   🕐 Hours to expiry: {hours_to_expiry:.1f}h | Capital: ${self.current_capital:,.2f}")
        
        return True

    def check_enhanced_exit_conditions(self, position: Dict, current_spy_price: float, 
                                     current_time: str) -> Tuple[bool, str]:
        """Enhanced exit conditions with multiple profit targets"""
        
        # Calculate time metrics
        entry_time = pd.to_datetime(position['entry_time'])
        current_dt = pd.to_datetime(current_time)
        time_held_hours = (current_dt - entry_time).total_seconds() / 3600
        remaining_hours = position['hours_to_expiry'] - time_held_hours
        
        # Calculate current option price
        volatility = max(15, position['vix_entry']) / 100
        current_option_price = self.simulate_realistic_option_pricing(
            position['strike'], position['option_type'], 
            current_spy_price, max(0.1, remaining_hours), volatility
        )
        
        # Calculate P&L
        pnl_pct = (current_option_price - position['entry_price']) / position['entry_price']
        
        # ENHANCED EXIT CONDITIONS
        
        # Multi-level profit taking
        if pnl_pct >= self.strategy.profit_target_home:
            return True, "PROFIT_HOME_RUN", current_option_price
        elif pnl_pct >= self.strategy.profit_target_main:
            return True, "PROFIT_MAIN_TARGET", current_option_price
        elif pnl_pct >= self.strategy.profit_target_quick and remaining_hours < 3.0:
            return True, "PROFIT_QUICK_TARGET", current_option_price
            
        # Enhanced stop loss
        if pnl_pct <= -self.strategy.stop_loss:
            return True, "STOP_LOSS", current_option_price
            
        # Time-based exits
        if remaining_hours <= 0.5:  # 30 minutes to close
            return True, "TIME_EXIT", current_option_price
        elif remaining_hours <= 1.0 and pnl_pct < -0.20:  # Cut losses near close
            return True, "TIME_STOP", current_option_price
            
        return False, "HOLD", current_option_price

    def close_enhanced_position(self, position_id: str, exit_price: float, 
                              exit_reason: str, exit_time: str, spy_price: float):
        """Close position with enhanced P&L tracking"""
        
        position = self.positions[position_id]
        
        # Calculate detailed P&L
        premium_received = position['contracts'] * exit_price * 100
        pnl = premium_received - position['premium_paid']
        pnl_pct = pnl / position['premium_paid']
        
        # Update capital and metrics
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
        
        # Record detailed trade
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
            'capital_after': self.current_capital,
            'hours_held': (pd.to_datetime(exit_time) - pd.to_datetime(position['entry_time'])).total_seconds() / 3600
        }
        
        self.trades.append(trade_record)
        position['status'] = 'CLOSED'
        
        # Calculate current win rate
        win_rate = self.winning_trades / max(1, self.winning_trades + self.losing_trades)
        
        logging.info(f"🔐 Closed {position_id}: {exit_reason}")
        logging.info(f"   💰 P&L: ${pnl:+.2f} ({pnl_pct:+.1%}) | Win Rate: {win_rate:.1%}")

    def run_optimized_combined_backtest(self, start_date: str, end_date: str):
        """Run the optimized combined backtest (A + B)"""
        
        logging.info("🚀 OPTIMIZED COMBINED HIGH FREQUENCY 0DTE BACKTEST")
        logging.info("="*65)
        logging.info("🔧 OPTIMIZATIONS APPLIED:")
        logging.info(f"   📈 Confidence threshold: {self.strategy.confidence_threshold} (vs 0.4)")
        logging.info(f"   💰 Option range: ${self.strategy.min_option_price}-${self.strategy.max_option_price} (vs $0.5-$3.0)")
        logging.info(f"   🎯 Profit targets: {self.strategy.profit_target_quick:.0%}, {self.strategy.profit_target_main:.0%}, {self.strategy.profit_target_home:.0%}")
        logging.info(f"   🛑 Stop loss: {self.strategy.stop_loss:.0%} (vs 50%)")
        logging.info(f"   ⚡ Max daily trades: {self.strategy.max_daily_trades} (vs 15)")
        
        logging.info(f"\n📅 EXTENDED TESTING PERIOD: {start_date} to {end_date}")
        logging.info(f"💰 Starting Capital: ${self.starting_capital:,.2f}")
        logging.info(f"🎯 TARGET: 8+ trades/day + 40%+ win rate")
        
        # Download extended data
        data = self.download_extended_data(start_date, end_date)
        if data is None:
            logging.error("❌ Failed to download data")
            return
            
        # Process by trading day
        trading_days = data.groupby('Date')
        total_days = len(trading_days)
        
        for day_num, (date, day_data) in enumerate(trading_days, 1):
            logging.info(f"\n📅 Processing Day {day_num}/{total_days}: {date}")
            
            day_start_capital = self.current_capital
            day_trades = 0
            day_signals = 0
            
            # Process every 5-minute interval during core hours
            core_start = pd.to_datetime('10:00:00').time()
            core_end = pd.to_datetime('15:30:00').time()
            
            day_data_core = day_data[
                (day_data['Time'] >= core_start) & 
                (day_data['Time'] <= core_end)
            ]
            
            # Sample every 5 minutes for analysis
            intervals = range(0, len(day_data_core), 5)
            
            for i in intervals:
                if i >= len(day_data_core):
                    break
                    
                row = day_data_core.iloc[i]
                spy_price = row['Close']
                vix_level = row['VIX']
                timestamp = row['Datetime'].strftime('%Y-%m-%d %H:%M:%S')
                
                # Check exit conditions for open positions
                for pos_id, position in list(self.positions.items()):
                    if position['status'] == 'OPEN':
                        should_exit, exit_reason, exit_price = self.check_enhanced_exit_conditions(
                            position, spy_price, timestamp
                        )
                        if should_exit:
                            self.close_enhanced_position(pos_id, exit_price, exit_reason, timestamp, spy_price)
                
                # Market analysis with optimization
                current_data = day_data_core.iloc[:i+1]
                analysis = self.strategy.analyze_market_conditions(
                    spy_price, vix_level, timestamp, current_data
                )
                
                if analysis['signal'] != 'HOLD':
                    day_signals += 1
                    
                    logging.info(f"🎯 OPTIMIZED Signal: {analysis['signal']}, Confidence: {analysis['confidence']:.2f}")
                    logging.info(f"   📊 Net Factors - Bullish: {analysis['net_bullish']:.1f}, Bearish: {analysis['net_bearish']:.1f}")
                    logging.info(f"   📈 RSI: {analysis['rsi']:.1f}, VIX: {analysis['vix']:.1f}, Volume Surge: {analysis['volume_surge']}")
                    
                    # Execute optimized trade
                    if self.execute_optimized_trade(analysis['signal'], spy_price, vix_level, timestamp, analysis):
                        day_trades += 1
                        
            # Force close all positions at end of day
            for pos_id, position in list(self.positions.items()):
                if position['status'] == 'OPEN':
                    final_time = day_data.iloc[-1]['Datetime'].strftime('%Y-%m-%d %H:%M:%S')
                    final_spy = day_data.iloc[-1]['Close']
                    self.close_enhanced_position(pos_id, 0.01, "EOD_EXPIRATION", final_time, final_spy)
            
            # Day summary with enhanced metrics
            day_pnl = self.current_capital - day_start_capital
            self.daily_pnl[str(date)] = day_pnl
            self.daily_returns.append(day_pnl / day_start_capital)
            
            # Current win rate
            current_win_rate = self.winning_trades / max(1, self.total_trades)
            
            logging.info(f"📊 Day {day_num} Summary:")
            logging.info(f"   💰 Trades: {day_trades} | Signals: {day_signals}")
            logging.info(f"   📈 Daily P&L: ${day_pnl:+.2f}")
            logging.info(f"   🏆 Win Rate: {current_win_rate:.1%} ({self.winning_trades}/{self.total_trades})")
            logging.info(f"   💵 Capital: ${self.current_capital:,.2f}")
            
        # Final comprehensive results
        self.print_comprehensive_results()
        self.save_comprehensive_results()

    def print_comprehensive_results(self):
        """Print comprehensive results focusing on win rate and P&L"""
        
        total_return = (self.current_capital - self.starting_capital) / self.starting_capital
        trading_days = len(self.daily_pnl)
        trades_per_day = self.total_trades / trading_days if trading_days > 0 else 0
        win_rate = self.winning_trades / max(1, self.total_trades)
        
        logging.info("\n" + "="*80)
        logging.info("🎯 OPTIMIZED COMBINED BACKTEST RESULTS")
        logging.info("="*80)
        
        # Performance Overview
        logging.info("📊 PERFORMANCE OVERVIEW:")
        logging.info(f"   💰 Starting Capital: ${self.starting_capital:,.2f}")
        logging.info(f"   💰 Final Capital: ${self.current_capital:,.2f}")
        logging.info(f"   📈 Total Return: {total_return:.2%}")
        logging.info(f"   💵 Total P&L: ${self.total_pnl:+,.2f}")
        logging.info(f"   📅 Trading Days: {trading_days}")
        
        # Trading Frequency Analysis
        logging.info(f"\n⚡ TRADING FREQUENCY ANALYSIS:")
        logging.info(f"   🎯 Total Trades: {self.total_trades}")
        logging.info(f"   📊 Trades per Day: {trades_per_day:.1f}")
        
        if trades_per_day >= 8.0:
            logging.info("   ✅ FREQUENCY TARGET ACHIEVED! (8+ trades/day)")
        elif trades_per_day >= 6.0:
            logging.info(f"   🟡 Close to target ({trades_per_day:.1f} vs 8.0)")
        else:
            logging.info(f"   ⚠️ Below frequency target ({trades_per_day:.1f} vs 8.0)")
            
        # Win Rate Analysis (KEY FOCUS)
        logging.info(f"\n🏆 WIN RATE ANALYSIS (KEY METRIC):")
        logging.info(f"   ✅ Win Rate: {win_rate:.1%} ({self.winning_trades}/{self.total_trades})")
        
        if win_rate >= 0.50:
            logging.info("   🎉 EXCELLENT WIN RATE! (50%+)")
        elif win_rate >= 0.40:
            logging.info("   ✅ GOOD WIN RATE! (40%+)")
        elif win_rate >= 0.30:
            logging.info("   🟡 Acceptable win rate (30%+)")
        else:
            logging.info("   ⚠️ Win rate needs improvement (<30%)")
            
        # P&L Analysis
        if len(self.trades) > 0:
            df_trades = pd.DataFrame(self.trades)
            
            winning_trades = df_trades[df_trades['pnl'] > 0]
            losing_trades = df_trades[df_trades['pnl'] < 0]
            
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            
            logging.info(f"\n💰 P&L BREAKDOWN:")
            logging.info(f"   💚 Average Win: ${avg_win:.2f}")
            logging.info(f"   💔 Average Loss: ${avg_loss:.2f}")
            
            if avg_loss != 0:
                profit_factor = abs(avg_win * len(winning_trades)) / abs(avg_loss * len(losing_trades))
                logging.info(f"   📈 Profit Factor: {profit_factor:.2f}")
                
                if profit_factor >= 1.5:
                    logging.info("   ✅ EXCELLENT profit factor!")
                elif profit_factor >= 1.2:
                    logging.info("   ✅ Good profit factor")
                elif profit_factor >= 1.0:
                    logging.info("   🟡 Break-even profit factor")
                else:
                    logging.info("   ⚠️ Negative profit factor")
                    
        # Risk Metrics
        logging.info(f"\n🛡️ RISK METRICS:")
        logging.info(f"   📉 Max Drawdown: {self.max_drawdown:.2%}")
        
        if len(self.daily_returns) > 1:
            daily_returns_array = np.array(self.daily_returns)
            sharpe_ratio = (np.mean(daily_returns_array) / np.std(daily_returns_array) * 
                          np.sqrt(252)) if np.std(daily_returns_array) > 0 else 0
            logging.info(f"   📊 Sharpe Ratio: {sharpe_ratio:.2f}")
            
        # Exit Reason Analysis
        if len(self.trades) > 0:
            df_trades = pd.DataFrame(self.trades)
            exit_counts = df_trades['exit_reason'].value_counts()
            
            logging.info(f"\n🚪 EXIT ANALYSIS:")
            for exit_reason, count in exit_counts.items():
                pct = count / len(df_trades) * 100
                exit_pnl = df_trades[df_trades['exit_reason'] == exit_reason]['pnl'].sum()
                logging.info(f"   {exit_reason}: {count} trades ({pct:.1f}%), ${exit_pnl:+.2f} P&L")
                
        # Optimization Impact Summary
        logging.info(f"\n🔧 OPTIMIZATION IMPACT:")
        logging.info(f"   📈 Signal threshold lowered to {self.strategy.confidence_threshold}")
        logging.info(f"   💰 Option range expanded to ${self.strategy.min_option_price}-${self.strategy.max_option_price}")
        logging.info(f"   🎯 Multi-level profit targets implemented")
        logging.info(f"   🛑 Tighter stop loss at {self.strategy.stop_loss:.0%}")
        
        # Overall Assessment
        logging.info(f"\n🎯 OVERALL ASSESSMENT:")
        if trades_per_day >= 8.0 and win_rate >= 0.40:
            logging.info("   🎉 BOTH TARGETS ACHIEVED! Strategy ready for live trading.")
        elif trades_per_day >= 6.0 and win_rate >= 0.35:
            logging.info("   ✅ Strong performance, minor tweaks recommended.")
        else:
            logging.info("   🔧 Further optimization recommended before live trading.")

    def save_comprehensive_results(self):
        """Save comprehensive results to CSV files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed trades
        if len(self.trades) > 0:
            df_trades = pd.DataFrame(self.trades)
            filename = f"optimized_combined_trades_{timestamp}.csv"
            df_trades.to_csv(filename, index=False)
            logging.info(f"💾 Detailed trades saved to: {filename}")
            
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
                    'capital': self.starting_capital + cumulative_pnl,
                    'return_pct': (cumulative_pnl / self.starting_capital) * 100
                })
                
            df_daily = pd.DataFrame(daily_summary)
            filename = f"optimized_combined_daily_{timestamp}.csv"
            df_daily.to_csv(filename, index=False)
            logging.info(f"💾 Daily summary saved to: {filename}")
            
        logging.info("="*80)

if __name__ == "__main__":
    # Run the optimized combined backtest
    logging.info("🚀 Starting Optimized Combined Backtest (A + B)")
    
    backtest = OptimizedCombinedBacktest(starting_capital=25000)
    
    # Use shorter period due to Yahoo Finance 1m data limitations (max 8 days)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')  # 7 days = ~5 trading days
    
    logging.info(f"📅 Adjusted for Yahoo Finance limits: {start_date} to {end_date}")
    
    backtest.run_optimized_combined_backtest(start_date, end_date) 