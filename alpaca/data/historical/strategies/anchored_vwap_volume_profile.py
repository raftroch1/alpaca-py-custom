"""
Anchored VWAP Volume Profile Strategy

Refactored to match Alpaca-py SDK research boilerplate:
- Uses base strategy template structure
- Ready for integration with ThetaData and Backtrader
- No hardcoded API keys
- Type hints and docstrings included

TODO:
- Integrate with ThetaDataConnector for real option data
- Implement Backtrader-compatible logic for backtesting
- Add unit and integration tests
"""

# PATCH: Try direct relative import first, fallback to absolute if needed
try:
    from .templates.base_theta_strategy import BaseThetaStrategy
except ImportError:
    from alpaca.data.historical.strategies.templates.base_theta_strategy import BaseThetaStrategy

# PATCH: Import proven ThetaData connector
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'thetadata'))
from connector import ThetaDataConnector
from typing import Dict, Optional, Any, List
import pandas as pd
import numpy as np
import logging

class AnchoredVWAPVolumeProfileStrategy(BaseThetaStrategy):
    """
    Anchored VWAP + Volume Profile Options Strategy
    Inherits from BaseThetaStrategy for Alpaca-py research boilerplate.
    """
    def __init__(self, 
                 version: str = "v1",
                 starting_capital: float = 25000,
                 max_risk_per_trade: float = 0.02,
                 target_profit_per_trade: float = 0.04):
        super().__init__(
            strategy_name="anchored_vwap_volume_profile",
            version=version,
            starting_capital=starting_capital,
            max_risk_per_trade=max_risk_per_trade,
            target_profit_per_trade=target_profit_per_trade
        )
        # Initialize proven ThetaData connector (fixes 474 error)
        self.theta_connector = ThetaDataConnector()
        
        # Strategy parameters
        self.timeframe = '1Min'
        self.lookback_periods = 500
        self.atr_period = 14
        self.rsi_period = 14
        self.ema_fast = 5
        self.ema_slow = 9
        # Anchor point settings
        self.anchor_point: Optional[int] = None
        self.anchored_vwap: Optional[pd.Series] = None
        self.volume_profile: Dict = {}
        # Current positions
        self.positions: Dict[str, Any] = {}

    def calculate_anchored_vwap(self, df: pd.DataFrame, anchor_idx: int) -> pd.Series:
        """Calculate anchored VWAP from anchor point."""
        if anchor_idx >= len(df):
            return pd.Series(index=df.index)
        anchor_data = df.iloc[anchor_idx:].copy()
        typical_price = (anchor_data['high'] + anchor_data['low'] + anchor_data['close']) / 3
        vwap_numerator = (typical_price * anchor_data['volume']).cumsum()
        vwap_denominator = anchor_data['volume'].cumsum()
        anchored_vwap = vwap_numerator / vwap_denominator
        result = pd.Series(index=df.index, dtype=float)
        result.iloc[anchor_idx:] = anchored_vwap
        return result

    def calculate_volume_profile(self, df: pd.DataFrame, anchor_idx: int, price_bins: int = 50) -> Dict:
        """Calculate volume profile from anchor point."""
        if anchor_idx >= len(df):
            return {}
        anchor_data = df.iloc[anchor_idx:].copy()
        price_min = anchor_data['low'].min()
        price_max = anchor_data['high'].max()
        price_bins_edges = np.linspace(price_min, price_max, price_bins + 1)
        volume_at_price = {}
        for _, row in anchor_data.iterrows():
            price_range = np.linspace(row['low'], row['high'], 10)
            volume_per_price = row['volume'] / len(price_range)
            for price in price_range:
                bin_idx = np.digitize(price, price_bins_edges) - 1
                if 0 <= bin_idx < len(price_bins_edges) - 1:
                    price_level = (price_bins_edges[bin_idx] + price_bins_edges[bin_idx + 1]) / 2
                    volume_at_price[price_level] = volume_at_price.get(price_level, 0) + volume_per_price
        sorted_levels = sorted(volume_at_price.items(), key=lambda x: x[1], reverse=True)
        total_volume = sum(volume_at_price.values())
        value_area_volume = total_volume * 0.68
        poc_price = sorted_levels[0][0] if sorted_levels else price_min
        cumulative_volume = 0
        value_area_prices = []
        for price, volume in sorted_levels:
            cumulative_volume += volume
            value_area_prices.append(price)
            if cumulative_volume >= value_area_volume:
                break
        vah = max(value_area_prices) if value_area_prices else price_max
        val = min(value_area_prices) if value_area_prices else price_min
        return {
            'poc': poc_price,
            'vah': vah,
            'val': val,
            'volume_profile': volume_at_price
        }

    def analyze_market_conditions(self, spy_price: float, vix_level: float, date: str) -> Dict[str, Any]:
        """
        Analyze market conditions using Anchored VWAP and volume profile logic.
        Args:
            spy_price: Current SPY price
            vix_level: Current VIX level
            date: Current date (YYYY-MM-DD)
        Returns:
            Dictionary with analysis results and signals
        """
        # Fetch SPY data for lookback period
        df = self.get_spy_data(
            start_date=(pd.to_datetime(date) - pd.Timedelta(days=self.lookback_periods)).strftime('%Y-%m-%d'),
            end_date=date
        )
        if df is None or df.empty:
            self.logger.warning("No SPY data available for analysis.")
            return {'signal': 'HOLD', 'confidence': 0}
        
        # Find anchor point
        anchor_idx = self.identify_anchor_point(df)
        if anchor_idx is None:
            self.logger.info("No anchor point found, holding position.")
            return {'signal': 'HOLD', 'confidence': 0}
        
        # Calculate technical indicators
        anchored_vwap = self.calculate_anchored_vwap(df, anchor_idx)
        volume_profile = self.calculate_volume_profile(df, anchor_idx)
        atr = self.calculate_atr(df)
        
        current_price = df['close'].iloc[-1]
        current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else current_price * 0.02
        
        # Extract volume profile levels
        poc = volume_profile.get('poc', current_price)
        vah = volume_profile.get('vah', current_price) 
        val = volume_profile.get('val', current_price)
        vwap_value = anchored_vwap.iloc[-1] if not pd.isna(anchored_vwap.iloc[-1]) else current_price
        
        # Dynamic confluence analysis
        confluence_levels = [poc, vah, val, vwap_value]
        atr_threshold = current_atr * 0.5  # More dynamic threshold
        
        # Calculate confluence score with weighted importance
        confluence_score = 0
        level_proximity = {}
        
        weights = {'poc': 3.0, 'vah': 2.0, 'val': 2.0, 'vwap': 2.5}
        level_names = ['poc', 'vah', 'val', 'vwap']
        
        for i, level in enumerate(confluence_levels):
            distance = abs(current_price - level)
            level_name = level_names[i]
            level_proximity[level_name] = distance
            
            if distance <= atr_threshold:
                confluence_score += weights[level_name]
        
        # Price action context
        price_vs_vwap = "above" if current_price > vwap_value else "below"
        vwap_slope = self.calculate_vwap_slope(anchored_vwap)
        
        # VIX regime consideration
        vix_regime = "low" if vix_level < 20 else "high" if vix_level > 30 else "neutral"
        
        # Volume analysis
        recent_volume = df['volume'].iloc[-5:].mean()
        avg_volume = df['volume'].iloc[-20:].mean()
        volume_surge = recent_volume > avg_volume * 1.3
        
        # Enhanced signal generation
        signal = 'HOLD'
        confidence = 0
        
        # Minimum confluence threshold
        min_confluence = 4.0
        
        if confluence_score >= min_confluence:
            # Determine directional bias
            bullish_factors = 0
            bearish_factors = 0
            
            # VWAP trend analysis
            if vwap_slope > 0.1:
                bullish_factors += 1
            elif vwap_slope < -0.1:
                bearish_factors += 1
            
            # Price vs VWAP
            if current_price > vwap_value:
                bullish_factors += 1
            else:
                bearish_factors += 1
            
            # Volume confirmation
            if volume_surge:
                if current_price > vwap_value:
                    bullish_factors += 1
                else:
                    bearish_factors += 1
            
            # VIX regime consideration
            if vix_regime == "low" and current_price > vwap_value:
                bullish_factors += 0.5
            elif vix_regime == "high" and current_price < vwap_value:
                bearish_factors += 0.5
            
            # Generate signal based on factor analysis
            if bullish_factors > bearish_factors and bullish_factors >= 2:
                signal = 'BUY_CALL'
                confidence = min((bullish_factors + confluence_score/10) / 4, 1.0)
            elif bearish_factors > bullish_factors and bearish_factors >= 2:
                signal = 'BUY_PUT'
                confidence = min((bearish_factors + confluence_score/10) / 4, 1.0)
        
        analysis_result = {
            'signal': signal,
            'confidence': confidence,
            'current_price': current_price,
            'vwap': vwap_value,
            'vwap_slope': vwap_slope,
            'poc': poc,
            'vah': vah,
            'val': val,
            'confluence_score': confluence_score,
            'atr': current_atr,
            'vix_level': vix_level,
            'vix_regime': vix_regime,
            'volume_surge': volume_surge,
            'price_vs_vwap': price_vs_vwap,
            'anchor_idx': anchor_idx,
            'level_proximity': level_proximity
        }
        
        if signal != 'HOLD':
            self.logger.info(f"Signal: {signal}, Confidence: {confidence:.2f}, "
                           f"Confluence: {confluence_score:.1f}, VIX: {vix_level:.1f}")
        
        return analysis_result



    def execute_strategy(self, market_analysis: Dict[str, Any], spy_price: float, date: str) -> Optional[Dict[str, Any]]:
        """
        Execute the strategy based on market analysis.
        Args:
            market_analysis: Output from analyze_market_conditions
            spy_price: Current SPY price
            date: Current date (YYYY-MM-DD)
        Returns:
            Trade execution details or None
        """
        signal = market_analysis.get('signal', 'HOLD')
        confidence = market_analysis.get('confidence', 0)
        if signal == 'HOLD' or confidence < 0.6:
            self.logger.info("No actionable signal.")
            return None
        # Use ThetaData to get real option price - no fallback
        # For 0DTE trading: expiration = same day as trade
        exp_date_dt = pd.to_datetime(date)  # Same day expiration for 0DTE
        exp_str = exp_date_dt.strftime('%Y-%m-%d')  # Format for connector
        right = 'C' if signal == 'BUY_CALL' else 'P'
        # Round to nearest $5 strike (how SPY options are actually listed)
        strike = self.round_to_option_strike(spy_price)
        
        # Use proven ThetaData connector (fixes 474 error)  
        # Pass EXPIRATION date (same as trade date for 0DTE), not trade date
        self.logger.info(f"🔍 Requesting 0DTE option: SPY {exp_str} {strike} {right} (rounded from {spy_price:.2f})")
        option_price = self.theta_connector.get_option_price('SPY', exp_str, strike, right)
        self.logger.info(f"💰 ThetaData returned: {option_price}")
        
        if option_price is None:
            self.logger.warning("No option price available, skipping trade.")
            return None
        contracts = self.calculate_position_size(signal, option_price)
        trade = {
            'date': date,
            'signal': signal,
            'contracts': contracts,
            'option_price': option_price,
            'strike': strike,
            'right': right,
            'exp_date': exp_date_dt.strftime('%Y%m%d'),  # Standard YYYYMMDD format for display
            'confidence': confidence
        }
        self.logger.info(f"Executing trade: {trade}")
        return trade

    def calculate_position_size(self, strategy_type: str, premium_collected: float) -> int:
        """
        Calculate position size based on risk management.
        Args:
            strategy_type: Type of strategy/signal
            premium_collected: Option premium per contract
        Returns:
            Number of contracts to trade
        """
        max_risk = self.current_capital * self.max_risk_per_trade
        if premium_collected <= 0:
            return 0
        contracts = int(max_risk / (premium_collected * 100))  # 100 shares per contract
        return max(1, min(contracts, 10))

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range for volatility measurement."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def detect_pivot_highs_lows(self, df: pd.DataFrame, window: int = 5) -> Dict[str, List[int]]:
        """Detect swing highs and lows using local extrema."""
        highs = []
        lows = []
        
        for i in range(window, len(df) - window):
            # Check for swing high
            current_high = df['high'].iloc[i]
            is_swing_high = True
            for j in range(i - window, i + window + 1):
                if j != i and df['high'].iloc[j] >= current_high:
                    is_swing_high = False
                    break
            if is_swing_high:
                highs.append(i)
            
            # Check for swing low
            current_low = df['low'].iloc[i]
            is_swing_low = True
            for j in range(i - window, i + window + 1):
                if j != i and df['low'].iloc[j] <= current_low:
                    is_swing_low = False
                    break
            if is_swing_low:
                lows.append(i)
        
        return {'highs': highs, 'lows': lows}

    def detect_breakout_points(self, df: pd.DataFrame, lookback: int = 20) -> List[int]:
        """Detect significant breakout points."""
        breakouts = []
        
        for i in range(lookback, len(df)):
            current_close = df['close'].iloc[i]
            recent_high = df['high'].iloc[i-lookback:i].max()
            recent_low = df['low'].iloc[i-lookback:i].min()
            range_size = recent_high - recent_low
            
            # Breakout above recent high
            if current_close > recent_high and range_size > 0:
                volume_surge = df['volume'].iloc[i] > df['volume'].iloc[i-10:i].mean() * 1.5
                if volume_surge:
                    breakouts.append(i)
            
            # Breakdown below recent low  
            elif current_close < recent_low and range_size > 0:
                volume_surge = df['volume'].iloc[i] > df['volume'].iloc[i-10:i].mean() * 1.5
                if volume_surge:
                    breakouts.append(i)
        
        return breakouts

    def detect_volatility_spikes(self, df: pd.DataFrame) -> List[int]:
        """Detect significant volatility expansion points."""
        spikes = []
        atr = self.calculate_atr(df)
        atr_ma = atr.rolling(window=20).mean()
        
        for i in range(20, len(df)):
            if atr.iloc[i] > atr_ma.iloc[i] * 2.0:  # ATR spike
                spikes.append(i)
        
        return spikes

    def identify_anchor_point(self, df: pd.DataFrame) -> Optional[int]:
        """
        Identify significant anchor point using multiple criteria:
        1. Recent swing highs/lows
        2. Breakout points with volume
        3. Volatility spikes
        
        Args:
            df: DataFrame of price data
        Returns:
            Index of most significant recent anchor point or None
        """
        if len(df) < 50:
            return None
        
        # Look for anchor points in the last 100 bars
        lookback_start = max(0, len(df) - 100)
        recent_df = df.iloc[lookback_start:].copy()
        recent_df.reset_index(drop=True, inplace=True)
        
        # Find different types of anchor points
        pivots = self.detect_pivot_highs_lows(recent_df, window=3)
        breakouts = self.detect_breakout_points(recent_df, lookback=20)
        vol_spikes = self.detect_volatility_spikes(recent_df)
        
        # Score potential anchor points
        anchor_candidates = {}
        
        # Score swing highs/lows (recent ones get higher scores)
        for high_idx in pivots['highs']:
            if high_idx >= 20:  # Must have enough history
                recency_score = (high_idx / len(recent_df)) * 100  # 0-100
                anchor_candidates[high_idx] = recency_score + 50  # Base score for pivots
        
        for low_idx in pivots['lows']:
            if low_idx >= 20:
                recency_score = (low_idx / len(recent_df)) * 100
                anchor_candidates[low_idx] = recency_score + 50
        
        # Score breakouts (higher priority)
        for breakout_idx in breakouts:
            if breakout_idx >= 20:
                recency_score = (breakout_idx / len(recent_df)) * 100
                anchor_candidates[breakout_idx] = recency_score + 75  # Higher base score
        
        # Score volatility spikes
        for spike_idx in vol_spikes:
            if spike_idx >= 20:
                recency_score = (spike_idx / len(recent_df)) * 100
                if spike_idx in anchor_candidates:
                    anchor_candidates[spike_idx] += 25  # Bonus for existing candidates
                else:
                    anchor_candidates[spike_idx] = recency_score + 30
        
        if not anchor_candidates:
            # Fallback: use most recent significant swing point
            all_pivots = pivots['highs'] + pivots['lows']
            if all_pivots:
                recent_pivot = max([p for p in all_pivots if p >= 20])
                return lookback_start + recent_pivot
            return None
        
        # Return the highest scoring anchor point
        best_anchor_idx = max(anchor_candidates.keys(), key=lambda x: anchor_candidates[x])
        actual_index = lookback_start + best_anchor_idx
        
        self.logger.info(f"Found anchor point at index {actual_index} (score: {anchor_candidates[best_anchor_idx]:.1f})")
        return actual_index

    def calculate_vwap_slope(self, vwap_series: pd.Series, periods: int = 5) -> float:
        """Calculate the slope/trend of VWAP over recent periods."""
        if len(vwap_series) < periods + 1:
            return 0.0
        
        recent_vwap = vwap_series.dropna().tail(periods + 1)
        if len(recent_vwap) < 2:
            return 0.0
        
        # Simple slope calculation
        y_values = recent_vwap.values
        x_values = np.arange(len(y_values))
        
        # Linear regression slope
        slope = np.polyfit(x_values, y_values, 1)[0]
        return slope

    def round_to_option_strike(self, price: float, increment: float = 5.0) -> float:
        """Round price to nearest option strike increment (usually $5 for SPY)."""
        return round(price / increment) * increment

# Example usage (for development/testing only)
if __name__ == "__main__":
    strategy = AnchoredVWAPVolumeProfileStrategy()
    # TODO: Add test run or integration with backtest engine 