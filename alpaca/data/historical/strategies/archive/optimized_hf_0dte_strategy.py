"""
OPTIMIZED HIGH FREQUENCY 0DTE OPTIONS STRATEGY
==============================================
Enhanced version focusing on WIN RATE and P&L performance

Key Optimizations:
- Expanded option price range: $0.30-$5.00
- Lower confidence threshold: 0.35 (from 0.4)
- Dynamic exit conditions vs time-only exits
- Improved strike selection algorithm
- Better position sizing for risk management
- Intraday profit taking and stop losses
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_theta_strategy import BaseThetaStrategy
import thetadata
from connector import ThetaClientConnector
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class OptimizedHighFrequency0DTEStrategy(BaseThetaStrategy):
    def __init__(self, api_key: str = None, secret_key: str = None, base_url: str = None):
        super().__init__(api_key, secret_key, base_url)
        
        # OPTIMIZED PARAMETERS for better win rate
        self.confidence_threshold = 0.35  # Lowered from 0.4 (12.5% more sensitive)
        self.factor_threshold = 1.3       # Lowered from 1.5 (13% more sensitive)
        
        # EXPANDED OPTION SELECTION
        self.min_option_price = 0.30      # Expanded from 0.50
        self.max_option_price = 5.00      # Expanded from 3.00
        self.min_delta = 0.15             # Minimum delta for liquidity
        self.max_delta = 0.45             # Maximum delta to avoid deep ITM
        
        # ENHANCED RISK MANAGEMENT
        self.max_risk_per_trade = 0.012   # 1.2% of account (vs 1.5%)
        self.max_daily_trades = 20        # Increased from 15
        self.max_position_size = 15       # Increased from 10
        
        # IMPROVED EXIT CONDITIONS
        self.profit_target_1 = 0.25       # 25% quick profit
        self.profit_target_2 = 0.50       # 50% main target  
        self.profit_target_3 = 1.00       # 100% home run
        self.stop_loss = 0.40             # 40% stop (tighter than 50%)
        self.time_decay_exit = 0.70       # Exit at 70% time decay
        
        # Technical indicators (more sensitive)
        self.rsi_period = 12              # Shorter from 14
        self.rsi_oversold = 25            # More sensitive from 30
        self.rsi_overbought = 75          # More sensitive from 70
        self.ema_fast = 8                 # Faster from 9
        self.ema_slow = 18                # Faster from 21
        self.volume_surge_threshold = 1.3 # Lower from 1.5
        
        self.today_trades = 0
        self.last_trade_date = None
        
        # Performance tracking
        self.daily_pnl = {}
        self.win_count = 0
        self.loss_count = 0

    def analyze_market_conditions(self, spy_price: float, vix_level: float, date: str, market_data: pd.DataFrame = None) -> dict:
        """
        OPTIMIZED market analysis with enhanced signal generation
        """
        # Reset daily trade count
        current_date = date.split()[0]
        if self.last_trade_date != current_date:
            self.today_trades = 0
            self.last_trade_date = current_date
            
        if self.today_trades >= self.max_daily_trades:
            return {'signal': 'HOLD', 'confidence': 0, 'reason': 'daily_limit_reached'}
            
        # Use pre-loaded market data if available
        if market_data is not None and len(market_data) > 0:
            df = market_data.tail(30).copy()
        else:
            end_date = pd.to_datetime(date)
            start_date = end_date - pd.Timedelta(days=30)
            df = self.get_spy_data(start_date.strftime('%Y-%m-%d'), date)
        
        if df is None or len(df) < 8:  # Reduced minimum from 10
            return {'signal': 'HOLD', 'confidence': 0}
            
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
        recent_volume = df['volume'].iloc[-2:].mean()  # Last 2 periods
        avg_volume = df['volume'].iloc[-8:].mean()     # Shorter baseline
        volume_surge = recent_volume > avg_volume * self.volume_surge_threshold
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # Price momentum (multiple timeframes)
        if len(df) >= 3:
            momentum_1 = (current_price / df['close'].iloc[-2] - 1) * 100
            momentum_3 = (current_price / df['close'].iloc[-4] - 1) * 100 if len(df) >= 4 else momentum_1
        else:
            momentum_1 = momentum_3 = 0
            
        # VIX analysis (enhanced)
        vix_fear = vix_level > 20
        vix_complacency = vix_level < 15
        vix_spike = vix_level > 25
        
        # ENHANCED SIGNAL CALCULATION
        bullish_factors = 0
        bearish_factors = 0
        
        # RSI factors (more granular)
        if current_rsi <= self.rsi_oversold:
            bullish_factors += 2.0
        elif current_rsi <= 35:
            bullish_factors += 1.0
        elif current_rsi >= self.rsi_overbought:
            bearish_factors += 2.0
        elif current_rsi >= 65:
            bearish_factors += 1.0
            
        # EMA factors (enhanced)
        if ema_bullish and ema_momentum > 0.2:
            bullish_factors += 1.5
        elif not ema_bullish and ema_momentum < -0.2:
            bearish_factors += 1.5
        elif ema_bullish:
            bullish_factors += 0.8
        else:
            bearish_factors += 0.8
            
        # Volume factors (more weight)
        if volume_surge:
            if momentum_1 > 0.1:
                bullish_factors += 1.0
            elif momentum_1 < -0.1:
                bearish_factors += 1.0
            else:
                # Volume surge with sideways = potential breakout
                bullish_factors += 0.5
                bearish_factors += 0.5
                
        # VIX factors (enhanced)
        if vix_spike and current_rsi < 40:
            bullish_factors += 1.5  # Fear + oversold = buy opportunity
        elif vix_complacency and current_rsi > 60:
            bearish_factors += 1.0  # Complacency + overbought = sell signal
        elif vix_fear:
            bearish_factors += 0.5
            
        # Momentum factors (new)
        if momentum_1 > 0.3 and momentum_3 > 0.5:
            bullish_factors += 1.0
        elif momentum_1 < -0.3 and momentum_3 < -0.5:
            bearish_factors += 1.0
            
        # Calculate signal strength
        net_bullish = bullish_factors - bearish_factors * 0.8  # Slight bullish bias
        net_bearish = bearish_factors - bullish_factors * 0.8
        
        # Generate signals with lower threshold
        signal = 'HOLD'
        confidence = 0
        
        if net_bullish >= self.factor_threshold:
            signal = 'BUY_CALL'
            confidence = min(0.95, 0.4 + (net_bullish - self.factor_threshold) * 0.2)
        elif net_bearish >= self.factor_threshold:
            signal = 'BUY_PUT' 
            confidence = min(0.95, 0.4 + (net_bearish - self.factor_threshold) * 0.2)
            
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
            'reason': f"Net: {net_bullish:.1f}B vs {net_bearish:.1f}Be"
        }

    def find_best_option(self, signal: str, spy_price: float, date: str) -> dict:
        """
        ENHANCED option selection with expanded price range and delta filtering
        """
        try:
            signal_type = 'call' if signal == 'BUY_CALL' else 'put'
            exp_date = pd.to_datetime(date).strftime('%Y%m%d')
            
            # Get option chain
            chain = self.theta_client.get_option_chain(
                root='SPY',
                exp=exp_date,
                start_date=date,
                end_date=date
            )
            
            if chain is None or len(chain) == 0:
                return None
                
            # Filter by option type
            options = chain[chain['option_type'] == signal_type.upper()].copy()
            if len(options) == 0:
                return None
                
            # ENHANCED FILTERING
            # 1. Price range filter (expanded)
            options = options[
                (options['close'] >= self.min_option_price) & 
                (options['close'] <= self.max_option_price)
            ]
            
            # 2. Delta filter for liquidity (new)
            if 'delta' in options.columns:
                options = options[
                    (abs(options['delta']) >= self.min_delta) & 
                    (abs(options['delta']) <= self.max_delta)
                ]
            
            # 3. Volume filter (enhanced)
            if 'volume' in options.columns:
                options = options[options['volume'] > 5]  # Minimum liquidity
                
            # 4. Bid-ask spread filter (new)
            if 'bid' in options.columns and 'ask' in options.columns:
                options['spread'] = options['ask'] - options['bid']
                options['spread_pct'] = options['spread'] / options['close']
                options = options[options['spread_pct'] <= 0.20]  # Max 20% spread
            
            if len(options) == 0:
                return None
                
            # IMPROVED SELECTION ALGORITHM
            # Score options based on multiple factors
            options['score'] = 0
            
            # Price score (prefer middle range)
            price_mid = (self.min_option_price + self.max_option_price) / 2
            options['price_score'] = 1 - abs(options['close'] - price_mid) / price_mid
            options['score'] += options['price_score'] * 0.3
            
            # Delta score (prefer around 0.30)
            if 'delta' in options.columns:
                target_delta = 0.30
                options['delta_score'] = 1 - abs(abs(options['delta']) - target_delta) / target_delta
                options['score'] += options['delta_score'] * 0.3
            
            # Volume score
            if 'volume' in options.columns:
                max_vol = options['volume'].max()
                if max_vol > 0:
                    options['volume_score'] = options['volume'] / max_vol
                    options['score'] += options['volume_score'] * 0.2
            
            # Spread score (lower spread = higher score)
            if 'spread_pct' in options.columns:
                max_spread = options['spread_pct'].max()
                if max_spread > 0:
                    options['spread_score'] = 1 - (options['spread_pct'] / max_spread)
                    options['score'] += options['spread_score'] * 0.2
            
            # Select best option
            best_option = options.loc[options['score'].idxmax()]
            
            return {
                'strike': best_option['strike'],
                'option_type': signal_type,
                'price': best_option['close'],
                'delta': best_option.get('delta', 0),
                'volume': best_option.get('volume', 0),
                'spread': best_option.get('spread', 0),
                'score': best_option['score']
            }
            
        except Exception as e:
            logging.error(f"Error finding option: {e}")
            return None

    def calculate_position_size(self, option_price: float, confidence: float, account_value: float = 25000) -> int:
        """
        ENHANCED position sizing with better risk management
        """
        # Base position calculation
        max_risk_amount = account_value * self.max_risk_per_trade
        base_contracts = max(1, int(max_risk_amount / (option_price * 100)))
        
        # Confidence adjustment (more aggressive)
        confidence_multiplier = 0.8 + (confidence * 0.6)  # 0.8 to 1.4x
        adjusted_contracts = int(base_contracts * confidence_multiplier)
        
        # Option price adjustment (favor cheaper options)
        if option_price < 1.0:
            price_multiplier = 1.3
        elif option_price < 2.0:
            price_multiplier = 1.1
        else:
            price_multiplier = 0.9
            
        final_contracts = int(adjusted_contracts * price_multiplier)
        
        # Apply limits
        final_contracts = min(final_contracts, self.max_position_size)
        final_contracts = max(final_contracts, 1)
        
        logging.info(f"📊 Position Sizing: Option=${option_price:.2f}, Confidence={confidence:.2f}")
        logging.info(f"   🔢 Base={base_contracts}, Adjusted={adjusted_contracts}, Risk={max_risk_amount:.0f}, Final={final_contracts}")
        
        return final_contracts

    def should_exit_position(self, entry_price: float, current_price: float, time_held: float, 
                           time_to_expiry: float) -> tuple:
        """
        ENHANCED exit conditions for better win rate
        """
        pnl_pct = (current_price - entry_price) / entry_price
        time_decay_pct = time_held / (time_held + time_to_expiry) if time_to_expiry > 0 else 1.0
        
        # Profit taking (multiple levels)
        if pnl_pct >= self.profit_target_3:
            return True, "PROFIT_TARGET_3", pnl_pct
        elif pnl_pct >= self.profit_target_2:
            return True, "PROFIT_TARGET_2", pnl_pct  
        elif pnl_pct >= self.profit_target_1 and time_decay_pct > 0.4:
            return True, "PROFIT_TARGET_1", pnl_pct
            
        # Stop loss
        if pnl_pct <= -self.stop_loss:
            return True, "STOP_LOSS", pnl_pct
            
        # Time decay exit
        if time_decay_pct >= self.time_decay_exit:
            return True, "TIME_DECAY", pnl_pct
            
        # End of day exit
        if time_to_expiry <= 0.5:  # 30 minutes to close
            return True, "TIME_EXIT", pnl_pct
            
        return False, "HOLD", pnl_pct
        
    def update_performance_metrics(self, pnl: float, exit_reason: str):
        """Track performance metrics"""
        if pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
            
        total_trades = self.win_count + self.loss_count
        win_rate = self.win_count / total_trades if total_trades > 0 else 0
        
        logging.info(f"📊 Performance Update: Win Rate: {win_rate:.1%} ({self.win_count}/{total_trades})") 