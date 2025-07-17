#!/usr/bin/env python3
"""
TRUE HIGH FREQUENCY 0DTE STRATEGY V1
Target: $250-$500 daily profit on $25K account (1-2% daily returns)
Approach: 15-25 trades per day with aggressive parameters

This strategy uses proven high-frequency methods:
- Ultra-sensitive signal generation 
- Rapid-fire intraday trading
- Multiple profit-taking levels
- Tight risk management
- Real market baseline + high-frequency simulation

Version: v1 - TRUE HIGH FREQUENCY
Author: Strategy Development Framework  
Date: 2025-01-17
"""

import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging
import warnings
warnings.filterwarnings('ignore')

# Import base template
sys.path.append(os.path.join(os.path.dirname(__file__), 'templates'))
from base_theta_strategy import BaseThetaStrategy


class TrueHighFrequency0DTEStrategy(BaseThetaStrategy):
    """
    TRUE HIGH FREQUENCY 0DTE Strategy
    
    Target: $250-$500 daily profit (1-2% returns)
    Method: 15-25 trades per day with ultra-aggressive parameters
    """
    
    def __init__(self):
        super().__init__(
            strategy_name="True High Frequency 0DTE",
            version="v1",
            starting_capital=25000,
            max_risk_per_trade=0.008,  # 0.8% per trade (very aggressive)
            target_profit_per_trade=0.0008  # 0.08% per trade (many small wins)
        )
        
        # TRUE HIGH FREQUENCY PARAMETERS
        self.daily_profit_target = 350  # $350/day target (1.4% return)
        self.min_daily_trades = 15      # Minimum trades for true HF
        self.max_daily_trades = 35      # Maximum trades per day
        
        # ULTRA AGGRESSIVE SIGNAL PARAMETERS
        self.confidence_threshold = 0.25    # Very low threshold (more signals)
        self.factor_threshold = 0.8         # Very low threshold
        
        # EXPANDED OPTION SELECTION (key for frequency)
        self.min_option_price = 0.10        # Accept very cheap options
        self.max_option_price = 12.00       # Wide range
        
        # RAPID PROFIT TAKING (many small wins)
        self.profit_target_scalp = 0.15     # 15% quick scalp
        self.profit_target_quick = 0.25     # 25% quick profit  
        self.profit_target_main = 0.40      # 40% main target
        self.profit_target_home = 0.75      # 75% home run (not 100%+)
        self.stop_loss = 0.30               # 30% stop loss
        
        # ULTRA SENSITIVE TECHNICAL PARAMETERS
        self.rsi_period = 8                 # Very short period
        self.rsi_oversold = 32              # Less extreme
        self.rsi_overbought = 68            # Less extreme  
        self.ema_fast = 5                   # Very fast
        self.ema_slow = 12                  # Fast
        
        # HIGH FREQUENCY SAMPLING
        self.sampling_interval = 2          # Sample every 2 minutes
        self.min_data_points = 5            # Minimal history needed
        
        # POSITION SIZING (aggressive for daily target)
        self.max_position_size = 25         # Larger positions
        self.base_contracts = 3             # Start with 3 contracts minimum
        
        # TRACKING
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_date = None
        self.open_positions = {}
        self.position_counter = 0
        self.trades = []
        
        # Additional tracking attributes
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.current_capital = self.starting_capital

    def run_backtest(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Run TRUE HIGH FREQUENCY backtest targeting $250-500 daily profit
        """
        self.logger.info("🚀 TRUE HIGH FREQUENCY 0DTE BACKTEST - $250-500/DAY TARGET")
        self.logger.info(f"🎯 Daily Target: ${self.daily_profit_target}")
        self.logger.info(f"⚡ Target Frequency: {self.min_daily_trades}-{self.max_daily_trades} trades/day") 
        self.logger.info(f"📊 Account Size: ${self.starting_capital:,.2f}")
        
        # Get real market data baseline
        spy_data = self._get_market_baseline(start_date, end_date)
        if spy_data.empty:
            return self._create_empty_results()
            
        # Generate HIGH FREQUENCY intraday data
        hf_data = self._generate_true_hf_data(spy_data)
        
        # Process each trading day with ultra-high frequency
        self._process_hf_trading_days(hf_data)
        
        return self._generate_hf_results()

    def _get_market_baseline(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get real market data as baseline"""
        self.logger.info("📊 Fetching real market baseline...")
        
        start_str = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        end_str = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
        
        spy = yf.download('SPY', start=start_str, end=end_str, progress=False)
        vix = yf.download('^VIX', start=start_str, end=end_str, progress=False)
        
        if len(spy) == 0:
            self.logger.error("❌ No market data available")
            return pd.DataFrame()
            
        # Combine SPY and VIX
        market_data = spy.copy()
        market_data['VIX'] = vix['Close'] if len(vix) > 0 else 16.0
        
        self.logger.info(f"✅ Retrieved {len(market_data)} trading days")
        return market_data

    def _generate_true_hf_data(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate TRUE HIGH FREQUENCY minute-by-minute data
        
        Creates ultra-granular data for maximum trading opportunities
        """
        self.logger.info("⚡ Generating TRUE HIGH FREQUENCY intraday data...")
        
        hf_records = []
        
        for date, row in daily_data.iterrows():
            if date.weekday() >= 5:  # Skip weekends
                continue
                
            # Extract daily values
            open_price = float(row['Open'])
            close_price = float(row['Close']) 
            high_price = float(row['High'])
            low_price = float(row['Low'])
            volume = float(row['Volume'])
            vix_level = float(row['VIX']) if 'VIX' in row else 16.0
            
            # Generate EVERY MINUTE of trading day (390 minutes)
            current_price = open_price
            
            for minute in range(390):
                # Market time: 9:30 AM to 4:00 PM
                market_time = datetime.combine(
                    date.date(), 
                    datetime.min.time()
                ) + timedelta(minutes=570 + minute)
                
                # Realistic intraday price evolution
                progress = minute / 390
                
                # Target price based on daily close
                target = open_price + (close_price - open_price) * progress
                
                # Add realistic noise and momentum
                noise = np.random.normal(0, 0.0008)  # 0.08% noise per minute
                momentum = np.random.normal(0, 0.0015)  # Momentum component
                
                price_change = (target - current_price) * 0.15 + noise + momentum
                current_price *= (1 + price_change)
                
                # Keep within reasonable daily range
                current_price = max(low_price * 0.995, min(high_price * 1.005, current_price))
                
                # Realistic minute volume
                minute_volume = max(5000, np.random.normal(volume/390, volume/390 * 0.4))
                
                # VIX with intraday variation
                vix_noise = np.random.normal(0, vix_level * 0.015)
                minute_vix = max(10, min(40, vix_level + vix_noise))
                
                hf_records.append({
                    'datetime': market_time,
                    'date': date.date(),
                    'time': market_time.time(),
                    'close': current_price,
                    'high': current_price * (1 + abs(np.random.normal(0, 0.0003))),
                    'low': current_price * (1 - abs(np.random.normal(0, 0.0003))),
                    'volume': minute_volume,
                    'vix': minute_vix
                })
        
        df = pd.DataFrame(hf_records)
        self.logger.info(f"⚡ Generated {len(df):,} minute bars for TRUE HF analysis")
        
        return df

    def _process_hf_trading_days(self, hf_data: pd.DataFrame):
        """Process high-frequency data with ultra-aggressive trading"""
        
        trading_days = hf_data.groupby('date')
        
        for day_num, (date, day_data) in enumerate(trading_days, 1):
            self.logger.info(f"📅 HF Trading Day {day_num}: {date}")
            
            # Reset daily counters
            self._reset_daily_counters(str(date))
            day_start_capital = self.current_capital
            
            # Core trading hours (10:00 AM - 3:45 PM for maximum opportunity)
            core_data = day_data[
                (day_data['time'] >= pd.to_datetime('10:00:00').time()) &
                (day_data['time'] <= pd.to_datetime('15:45:00').time())
            ]
            
            if len(core_data) < 20:
                continue
                
            # ULTRA HIGH FREQUENCY SAMPLING (every 2 minutes)
            sample_indices = range(0, len(core_data), self.sampling_interval)
            
            for idx in sample_indices:
                if idx >= len(core_data):
                    break
                    
                row = core_data.iloc[idx]
                
                # Check exits first
                self._check_hf_exits(row['close'], str(date))
                
                # Stop if daily target hit
                if self.daily_pnl >= self.daily_profit_target:
                    self.logger.info(f"🎯 DAILY TARGET HIT: ${self.daily_pnl:.2f}")
                    break
                    
                # Stop if max trades reached
                if self.daily_trades >= self.max_daily_trades:
                    break
                    
                # Get market context (minimal history needed)
                context_data = core_data.iloc[:idx+1].tail(20)
                if len(context_data) < self.min_data_points:
                    continue
                    
                # ULTRA AGGRESSIVE signal analysis
                analysis = self._ultra_aggressive_analysis(
                    row['close'], row['vix'], context_data
                )
                
                if analysis['should_trade']:
                    # Execute with aggressive sizing
                    self._execute_hf_trade(analysis, row['close'], row['vix'], str(date))
                    
            # EOD cleanup
            self._close_all_eod()
            
            # Daily summary
            day_pnl = self.current_capital - day_start_capital
            win_rate = self.winning_trades / max(1, self.total_trades)
            
            self.logger.info(f"📊 Day {day_num} Summary:")
            self.logger.info(f"   ⚡ Trades: {self.daily_trades} (target: {self.min_daily_trades}+)")
            self.logger.info(f"   💰 P&L: ${day_pnl:+.2f} (target: ${self.daily_profit_target})")
            self.logger.info(f"   📈 Win Rate: {win_rate:.1%}")
            self.logger.info(f"   💵 Capital: ${self.current_capital:,.2f}")

    def _ultra_aggressive_analysis(self, spy_price: float, vix_level: float, 
                                 context_data: pd.DataFrame) -> Dict[str, Any]:
        """
        ULTRA AGGRESSIVE signal generation for TRUE HIGH FREQUENCY
        
        Uses very sensitive parameters to generate 15-25 signals per day
        """
        
        if len(context_data) < self.min_data_points:
            return {'should_trade': False, 'confidence': 0}
            
        prices = context_data['close']
        
        # Ultra-fast technical indicators
        rsi = self._fast_rsi(prices, self.rsi_period)
        ema_fast = prices.ewm(span=self.ema_fast).mean().iloc[-1]
        ema_slow = prices.ewm(span=self.ema_slow).mean().iloc[-1]
        
        # Price momentum (very sensitive)
        momentum_1min = (spy_price / prices.iloc[-2] - 1) * 100 if len(prices) >= 2 else 0
        momentum_3min = (spy_price / prices.iloc[-4] - 1) * 100 if len(prices) >= 4 else momentum_1min
        
        # Volume surge detection
        if 'volume' in context_data.columns:
            recent_vol = context_data['volume'].iloc[-3:].mean()
            avg_vol = context_data['volume'].mean()
            volume_surge = recent_vol > avg_vol * 1.1  # Very sensitive
        else:
            volume_surge = False
            
        # ULTRA AGGRESSIVE scoring system
        bullish_score = 0
        bearish_score = 0
        
        # RSI scoring (very granular)
        if rsi <= 30:
            bullish_score += 4.0
        elif rsi <= 35: 
            bullish_score += 3.0
        elif rsi <= 40:
            bullish_score += 2.0
        elif rsi <= 45:
            bullish_score += 1.0
            
        if rsi >= 70:
            bearish_score += 4.0
        elif rsi >= 65:
            bearish_score += 3.0  
        elif rsi >= 60:
            bearish_score += 2.0
        elif rsi >= 55:
            bearish_score += 1.0
            
        # EMA scoring
        if ema_fast > ema_slow:
            bullish_score += 2.0
        else:
            bearish_score += 2.0
            
        # Momentum scoring (very sensitive)
        if momentum_1min > 0.02:  # 0.02% threshold
            bullish_score += 3.0
        elif momentum_1min > 0.01:
            bullish_score += 2.0
        elif momentum_1min < -0.02:
            bearish_score += 3.0
        elif momentum_1min < -0.01:
            bearish_score += 2.0
            
        # VIX scoring
        if vix_level > 18 and rsi < 40:
            bullish_score += 2.5
        elif vix_level < 14 and rsi > 60:
            bearish_score += 2.5
            
        # Volume scoring
        if volume_surge:
            if momentum_1min > 0:
                bullish_score += 1.5
            else:
                bearish_score += 1.5
                
        # Net signal calculation
        net_bullish = bullish_score - bearish_score * 0.8
        net_bearish = bearish_score - bullish_score * 0.8
        
        # Generate signals (VERY LOW threshold for high frequency)
        should_trade = False
        signal_type = 'HOLD'
        confidence = 0
        
        if net_bullish >= self.factor_threshold:
            confidence = min(0.85, 0.3 + net_bullish * 0.1)
            if confidence >= self.confidence_threshold:
                should_trade = True
                signal_type = 'CALL_HF'
                
        elif net_bearish >= self.factor_threshold:
            confidence = min(0.85, 0.3 + net_bearish * 0.1)
            if confidence >= self.confidence_threshold:
                should_trade = True
                signal_type = 'PUT_HF'
        
        return {
            'should_trade': should_trade,
            'signal_type': signal_type,
            'confidence': confidence,
            'rsi': rsi,
            'momentum_1min': momentum_1min,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'net_bullish': net_bullish,
            'net_bearish': net_bearish
        }

    def _execute_hf_trade(self, analysis: Dict, spy_price: float, vix_level: float, date: str):
        """Execute high-frequency trade with aggressive position sizing"""
        
        # Generate strike (slightly OTM for cheaper premiums)
        if analysis['signal_type'] == 'CALL_HF':
            strike = round(spy_price + np.random.uniform(0.5, 2.5), 0)
            option_type = 'call'
        else:
            strike = round(spy_price - np.random.uniform(0.5, 2.5), 0)
            option_type = 'put'
            
        # Simulate option price
        option_price = self._simulate_hf_option_price(strike, option_type, spy_price, vix_level)
        
        # Check price range (expanded for HF)
        if option_price < self.min_option_price or option_price > self.max_option_price:
            return False
            
        # AGGRESSIVE position sizing for daily target
        confidence_mult = 0.6 + analysis['confidence'] * 0.8  # 0.6 to 1.4x
        base_risk = self.current_capital * self.max_risk_per_trade
        contracts = max(self.base_contracts, int(base_risk / (option_price * 100) * confidence_mult))
        contracts = min(contracts, self.max_position_size)
        
        total_cost = contracts * option_price * 100
        
        # Capital check
        if total_cost > self.current_capital * 0.15:  # Max 15% per trade
            return False
            
        # Execute trade
        self.position_counter += 1
        position_id = f"HF_{self.position_counter:04d}"
        
        position = {
            'id': position_id,
            'entry_date': date,
            'signal': analysis['signal_type'],
            'strike': strike,
            'option_type': option_type,
            'contracts': contracts,
            'entry_price': option_price,
            'total_cost': total_cost,
            'spy_entry': spy_price,
            'vix_entry': vix_level,
            'confidence': analysis['confidence']
        }
        
        self.open_positions[position_id] = position
        self.current_capital -= total_cost
        self.daily_trades += 1
        self.total_trades += 1
        
        self.logger.info(f"⚡ HF TRADE #{self.total_trades}: {analysis['signal_type']}")
        self.logger.info(f"   💰 {contracts} x {strike}{option_type[0].upper()} @ ${option_price:.2f} = ${total_cost:.0f}")
        self.logger.info(f"   📊 Confidence: {analysis['confidence']:.2f}, RSI: {analysis['rsi']:.1f}")
        
        return True

    def _check_hf_exits(self, spy_price: float, date: str):
        """Check exit conditions with RAPID profit taking"""
        
        for pos_id in list(self.open_positions.keys()):
            position = self.open_positions[pos_id]
            
            # Simulate current option price
            current_price = self._simulate_hf_option_price(
                position['strike'], position['option_type'], spy_price, 15.0
            )
            
            entry_price = position['entry_price']
            pnl_pct = (current_price - entry_price) / entry_price
            
            should_exit = False
            exit_reason = "HOLD"
            
            # RAPID profit taking for high frequency
            if pnl_pct >= self.profit_target_scalp:
                should_exit = True
                exit_reason = "SCALP_PROFIT"
            elif pnl_pct >= self.profit_target_quick:
                should_exit = True  
                exit_reason = "QUICK_PROFIT"
            elif pnl_pct >= self.profit_target_main:
                should_exit = True
                exit_reason = "MAIN_PROFIT"
            elif pnl_pct >= self.profit_target_home:
                should_exit = True
                exit_reason = "HOME_RUN"
            elif pnl_pct <= -self.stop_loss:
                should_exit = True
                exit_reason = "STOP_LOSS"
                
            if should_exit:
                self._close_hf_position(pos_id, current_price, exit_reason, spy_price)

    def _close_hf_position(self, position_id: str, exit_price: float, exit_reason: str, spy_price: float):
        """Close high-frequency position and track P&L"""
        
        position = self.open_positions[position_id]
        
        # Calculate P&L
        premium_received = position['contracts'] * exit_price * 100
        pnl = premium_received - position['total_cost']
        pnl_pct = pnl / position['total_cost']
        
        # Update capital and tracking
        self.current_capital += premium_received
        self.daily_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
            
        # Record trade
        trade_record = {
            'position_id': position_id,
            'signal': position['signal'],
            'entry_date': position['entry_date'],
            'exit_reason': exit_reason,
            'contracts': position['contracts'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'spy_entry': position['spy_entry'],
            'spy_exit': spy_price,
            'confidence': position['confidence']
        }
        
        self.trades.append(trade_record)
        del self.open_positions[position_id]
        
        self.logger.info(f"🔐 Closed {position_id}: {exit_reason}, P&L: ${pnl:+.2f} ({pnl_pct:+.1%})")

    def _close_all_eod(self):
        """Close all positions at end of day"""
        for pos_id in list(self.open_positions.keys()):
            self._close_hf_position(pos_id, 0.01, "EOD_EXPIRATION", 0)

    def _simulate_hf_option_price(self, strike: float, option_type: str, spy_price: float, vix_level: float) -> float:
        """Simulate realistic high-frequency option pricing"""
        
        # Time to expiry (assume intraday 0DTE)
        time_factor = np.random.uniform(0.1, 0.4)  # Random time remaining
        
        # Intrinsic value
        if option_type == 'call':
            intrinsic = max(0, spy_price - strike)
        else:
            intrinsic = max(0, strike - spy_price)
            
        # Time value with VIX influence
        moneyness = abs(spy_price - strike) / spy_price
        vol_factor = vix_level / 100 
        
        time_value = (0.3 + vol_factor * 2) * time_factor * (1.5 - moneyness * 2)
        time_value = max(0, time_value)
        
        option_price = intrinsic + time_value
        
        # Add market noise
        noise = np.random.normal(0, 0.05)
        option_price *= (1 + noise)
        
        return max(0.05, option_price)

    def _fast_rsi(self, prices: pd.Series, period: int = 8) -> float:
        """Fast RSI calculation for high frequency"""
        if len(prices) < period:
            return 50.0
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi) if not np.isnan(rsi) else 50.0

    def _reset_daily_counters(self, date: str):
        """Reset daily tracking variables"""
        if self.last_trade_date != date:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_trade_date = date

    def _generate_hf_results(self) -> Dict[str, Any]:
        """Generate comprehensive high-frequency results"""
        
        total_return = (self.current_capital - self.starting_capital) / self.starting_capital
        
        # Calculate daily metrics
        if len(self.trades) > 0:
            df_trades = pd.DataFrame(self.trades)
            trading_days = len(df_trades['entry_date'].unique())
            daily_trades = self.total_trades / max(1, trading_days)
            daily_pnl = (self.current_capital - self.starting_capital) / max(1, trading_days)
        else:
            daily_trades = 0
            daily_pnl = 0
            trading_days = 1
            
        win_rate = self.winning_trades / max(1, self.total_trades)
        
        self.logger.info("\n" + "="*70)
        self.logger.info("⚡ TRUE HIGH FREQUENCY 0DTE RESULTS")
        self.logger.info("="*70)
        self.logger.info(f"🎯 DAILY TARGET: ${self.daily_profit_target}")
        self.logger.info(f"💰 Actual Daily P&L: ${daily_pnl:.2f}")
        self.logger.info(f"⚡ Daily Trades: {daily_trades:.1f} (target: {self.min_daily_trades}+)")
        self.logger.info(f"📈 Win Rate: {win_rate:.1%}")
        self.logger.info(f"💵 Total Return: {total_return:.2%}")
        self.logger.info(f"🏦 Final Capital: ${self.current_capital:,.2f}")
        
        # Target assessment
        if daily_pnl >= 250:
            self.logger.info("🎉 DAILY TARGET ACHIEVED!")
        elif daily_pnl >= 150:
            self.logger.info("✅ Good progress toward target")
        else:
            self.logger.info("⚠️ Below daily target - needs optimization")
            
        if daily_trades >= self.min_daily_trades:
            self.logger.info("⚡ HIGH FREQUENCY TARGET ACHIEVED!")
        else:
            self.logger.info("⚠️ Below frequency target")
            
        # Save results
        self._save_hf_results()
        
        return {
            'total_trades': self.total_trades,
            'daily_trade_frequency': daily_trades,
            'win_rate': win_rate * 100,
            'total_pnl': self.current_capital - self.starting_capital,
            'daily_pnl': daily_pnl,
            'total_return_pct': total_return * 100,
            'final_capital': self.current_capital,
            'target_achieved': daily_pnl >= 250 and daily_trades >= self.min_daily_trades
        }

    def _save_hf_results(self):
        """Save detailed high-frequency results"""
        if len(self.trades) > 0:
            df = pd.DataFrame(self.trades)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"true_hf_0dte_trades_{timestamp}.csv"
            
            # Ensure logs directory exists
            logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            
            filepath = os.path.join(logs_dir, filename)
            df.to_csv(filepath, index=False)
            self.logger.info(f"💾 HF results saved: {filename}")

    def _create_empty_results(self) -> Dict[str, Any]:
        """Create empty results for error cases"""
        return {
            'total_trades': 0,
            'daily_trade_frequency': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'daily_pnl': 0,
            'total_return_pct': 0,
            'final_capital': self.starting_capital,
            'target_achieved': False
        }

    # Required abstract methods from BaseThetaStrategy
    def analyze_market_conditions(self, spy_price: float, vix_level: float, 
                                date: str, market_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analyze market conditions for trading signals.
        This is a wrapper around the ultra-aggressive analysis for compatibility.
        """
        if market_data is not None and len(market_data) >= self.min_data_points:
            return self._ultra_aggressive_analysis(spy_price, vix_level, market_data)
        else:
            return {'should_trade': False, 'confidence': 0, 'signal_type': 'HOLD'}

    def calculate_position_size(self, option_price: float, confidence: float, 
                              account_value: float = None) -> int:
        """
        Calculate position size based on confidence and risk parameters.
        """
        if account_value is None:
            account_value = self.current_capital
            
        # Aggressive sizing for high frequency
        confidence_mult = 0.6 + confidence * 0.8
        base_risk = account_value * self.max_risk_per_trade
        contracts = max(self.base_contracts, int(base_risk / (option_price * 100) * confidence_mult))
        contracts = min(contracts, self.max_position_size)
        
        return max(1, contracts)

    def execute_strategy(self, market_analysis: Dict[str, Any], spy_price: float, 
                        date: str) -> Dict[str, Any]:
        """
        Execute trading strategy based on market analysis.
        This is a wrapper around the HF trade execution for compatibility.
        """
        if not market_analysis.get('should_trade', False):
            return None
            
        signal_type = market_analysis.get('signal_type', 'HOLD')
        confidence = market_analysis.get('confidence', 0)
        
        # Generate option details
        if signal_type == 'CALL_HF':
            strike = round(spy_price + np.random.uniform(0.5, 2.5), 0)
            option_type = 'call'
        elif signal_type == 'PUT_HF':
            strike = round(spy_price - np.random.uniform(0.5, 2.5), 0)
            option_type = 'put'
        else:
            return None
            
        # Simulate option price
        option_price = self._simulate_hf_option_price(strike, option_type, spy_price, 16.0)
        
        # Check price range
        if option_price < self.min_option_price or option_price > self.max_option_price:
            return None
            
        # Calculate position size
        contracts = self.calculate_position_size(option_price, confidence)
        total_cost = contracts * option_price * 100
        
        # Return trade details
        return {
            'signal': signal_type,
            'strike': strike,
            'option_type': option_type,
            'contracts': contracts,
            'entry_price': option_price,
            'total_cost': total_cost,
            'confidence': confidence,
            'spy_price': spy_price,
            'date': date
        } 