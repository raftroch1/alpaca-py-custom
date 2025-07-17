# High Frequency 0DTE Options Strategy

## 📋 Overview

This is an enhanced **Anchored VWAP Volume Profile** strategy specifically designed for **high-frequency 0DTE (Zero Days to Expiration)** options trading. The strategy generates 8+ trades per day by using lower confidence thresholds and sophisticated technical analysis to identify short-term market opportunities.

### 🎯 Core Strategy Performance
- **Frequency**: 8.2 trades/day (vs original 0.18/day - **45x improvement**)
- **Target Options**: $0.50-$3.00 price range to avoid far OTM options
- **Win Rate**: 34.1% (needs optimization)
- **Profit Factor**: 0.67 (needs improvement to >1.0)
- **Risk Management**: 50% stop loss, 100% profit target
- **Position Sizing**: Smart contract allocation based on option price

## 🔧 How It Works - Core Mechanics

### 1. **Market Analysis Engine**
The strategy combines multiple technical indicators with lowered thresholds for higher frequency:

```python
# Enhanced Signal Generation (Lower Thresholds)
min_confidence = 0.4    # Lowered from 0.6 (33% easier to trigger)
min_factors = 1.5       # Lowered from 2.0 (25% easier to trigger)

# Technical Factors Analyzed:
- Anchored VWAP position vs current price
- RSI oversold/overbought (40/60 vs 30/70)
- Volume surge detection (1.2x vs 1.5x)
- Price momentum (0.2% threshold)
- VIX regime consideration
- EMA crossovers (3/7 periods)
```

### 2. **Option Selection Process**
Smart option filtering to avoid far OTM options:

```python
# Option Price Filtering
min_option_price = 0.50   # Minimum option value
max_option_price = 3.00   # Maximum option value
strike_increment = 5.0    # SPY $5 strike spacing

# Strike Selection Logic:
1. Round SPY price to nearest $5 strike
2. Test base strike + nearby strikes (±5, ±10)
3. Accept first option within $0.50-$3.00 range
4. Skip trade if no suitable options found
```

### 3. **Position Sizing Algorithm**
Dynamic contract allocation based on option price:

```python
# Smart Position Sizing
if option_price <= 1.00:
    base_contracts = 8      # Buy more cheap options
elif option_price <= 2.00:
    base_contracts = 5      # Medium allocation
else:
    base_contracts = 3      # Conservative on expensive options

# Risk-based limits: max 10 contracts, min 1 contract
```

### 4. **Risk Management System**
Comprehensive intraday risk controls:

```python
# Exit Conditions
stop_loss = 50%           # Automatic stop at 50% loss
profit_target = 100%      # Take profits at 100% gain
eod_expiration = True     # Close all positions at market close

# Exit Priority:
1. Stop Loss (24% of trades) → 0% win rate
2. Profit Target (27% of trades) → 100% win rate  
3. EOD Expiration (49% of trades) → 15% win rate
```

## 📊 Key Parameters & Configuration

### **Signal Generation Parameters**
| Parameter | Value | Purpose | Original |
|-----------|-------|---------|----------|
| `min_confidence` | 0.4 | Entry threshold | 0.6 |
| `min_factors` | 1.5 | Factor requirement | 2.0 |
| `rsi_oversold` | 40 | RSI buy signal | 30 |
| `rsi_overbought` | 60 | RSI sell signal | 70 |
| `volume_surge` | 1.2x | Volume threshold | 1.5x |
| `ema_fast` | 3 | Fast EMA period | 5 |
| `ema_slow` | 7 | Slow EMA period | 9 |

### **Option Filtering Parameters**
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `min_option_price` | $0.50 | Avoid penny options |
| `max_option_price` | $3.00 | Avoid expensive options |
| `strike_increment` | $5.00 | SPY option spacing |
| `expiration` | Same day | 0DTE requirement |

### **Risk Management Parameters**
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `stop_loss_pct` | 50% | Maximum loss tolerance |
| `profit_target_pct` | 100% | Profit taking level |
| `max_risk_per_trade` | 1.5% | Capital risk limit |
| `max_position_size` | 10 contracts | Position limit |

## 🔌 Technology Integration

### **ThetaData Integration**
Real option pricing without simulation fallbacks:

```python
# ThetaData Connector Usage
from connector import ThetaDataConnector

theta_connector = ThetaDataConnector()
option_price = theta_connector.get_option_price(
    symbol='SPY',
    exp_date='2024-07-15',  # Same day for 0DTE
    strike=590.0,
    right='C'  # Call or Put
)

# NO SIMULATION FALLBACKS - Real data only
```

### **Backtrader Integration**
The strategy is designed to work with Backtrader backtesting framework:

```python
# Backtrader Compatibility
class HighFrequency0DTEStrategy(BaseThetaStrategy):
    def next(self):
        # Market analysis
        analysis = self.analyze_market_conditions(self.data)
        
        # Trade execution  
        if analysis['signal'] != 'HOLD':
            trade = self.execute_strategy(analysis, self.spy_price, self.date)
```

### **Alpaca Integration**
Market data sourcing for SPY underlying:

```python
# Alpaca Data Client
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest

# Fetches 1-minute SPY bars for analysis
stock_client = StockHistoricalDataClient()
```

## 📁 File Structure & Usage

### **Core Strategy Files**

#### **For Live Trading:**
```bash
# Primary Strategy Files
├── high_frequency_0dte_strategy.py      # Main strategy class
├── templates/base_theta_strategy.py     # Base strategy template
└── thetadata/connector.py               # ThetaData integration

# Required Dependencies
├── alpaca/ (SDK for market data)
└── backtrader/ (for strategy framework)
```

#### **For Backtesting:**
```bash
# Backtesting Files
├── high_frequency_backtest.py           # Full backtest framework
├── simple_hf_demo.py                    # Quick demo/validation
├── detailed_analytics.py               # Performance analysis
└── logs/ (generated results)
    ├── high_frequency_trades_*.csv      # Trade logs
    └── daily_summary_*.csv              # Daily performance
```

### **How to Run**

#### **🚀 Live Trading Execution:**
```bash
# 1. Ensure ThetaData Terminal is running
# 2. Set up environment variables for Alpaca API keys
# 3. Run the strategy

cd alpaca/data/historical/strategies/
python high_frequency_0dte_strategy.py
```

#### **📊 Backtesting Execution:**
```bash
# Quick Demo (5 trading days)
python simple_hf_demo.py

# Full Backtest (with real market data)
python high_frequency_backtest.py

# Detailed Analytics
python detailed_analytics.py
```

#### **📈 Performance Analysis:**
```bash
# View comprehensive analytics
python detailed_analytics.py

# Check generated CSV files
ls logs/detailed_trades_*.csv
ls logs/daily_summary_*.csv
```

## 🎯 Current Performance Metrics

### **Achieved Objectives ✅**
- **High Frequency**: 8.2 trades/day (vs target 1-2)
- **Option Filtering**: 100% within $0.50-$3.00 range
- **Risk Management**: Automated stop losses & profit targets
- **Real Data Integration**: ThetaData + Alpaca APIs

### **Performance Summary**
```
📊 Strategy Overview:
- Total Trades: 41 (across 5 days)
- Win Rate: 34.1% (14 wins, 27 losses)
- Profit Factor: 0.67 (needs improvement)
- Max Win: +$1,044
- Max Loss: -$1,078
- Avg P&L/Trade: -$110.76

🎯 Exit Analysis:
- Profit Targets: 27% of trades (100% win rate)
- Stop Losses: 24% of trades (0% win rate) 
- EOD Expiration: 49% of trades (15% win rate)
```

## 🔧 Future Improvements

### **Priority 1: Improve Win Rate (34.1% → 45%+)**

#### **Entry Criteria Optimization**
```python
# Proposed Changes:
1. Increase confidence threshold: 0.4 → 0.5
2. Add momentum confirmation filters
3. Implement market regime detection
4. Add volatility clustering analysis
5. Volume profile confluence requirements

# Implementation:
- Add VIX spike detection
- Require multiple timeframe alignment
- Filter out low-volume periods
- Add support/resistance levels
```

#### **Signal Quality Enhancement**
```python
# Advanced Filters:
1. Options flow sentiment analysis
2. Market maker positioning indicators  
3. Intraday trend strength measurement
4. News/event filtering
5. Market microstructure analysis
```

### **Priority 2: Optimize Exit Strategy**

#### **Dynamic Exit Management**
```python
# Current Issue: 49% EOD expirations with 15% win rate
# Solutions:
1. Implement trailing stops
2. Add time-based exits (e.g., 2pm cutoff)
3. Volatility-based position sizing
4. Dynamic profit targets based on IV
5. Early exit on adverse momentum

# Proposed Exit Hierarchy:
1. 30-minute time stop (if losing)
2. Volatility expansion exits
3. Technical level breaches
4. Profit target scaling (25%, 50%, 100%)
```

### **Priority 3: Signal Type Optimization**

#### **CALL vs PUT Performance Analysis**
```python
# Current Results:
# CALL: 23 trades, -$228 P&L, 39.1% win rate  ✅ Better
# PUT: 18 trades, -$4,313 P&L, 27.8% win rate ❌ Worse

# Optimizations:
1. Bias toward CALL signals in low VIX environments
2. Improve PUT signal quality with put/call ratio
3. Market maker flow analysis for directional bias
4. Sector rotation impact on SPY direction
```

### **Priority 4: Risk Management Enhancement**

#### **Advanced Position Sizing**
```python
# Current: Fixed percentage risk model
# Proposed: Dynamic risk allocation

class AdvancedRiskManager:
    def calculate_position_size(self, option_price, iv, time_to_expiry, confidence):
        # Kelly Criterion integration
        # Volatility-adjusted sizing  
        # Confidence-weighted allocation
        # Portfolio heat monitoring
        return optimal_contracts
```

#### **Portfolio-Level Risk Controls**
```python
# Implement:
1. Maximum daily loss limits (-$500)
2. Maximum open positions (3 at once)
3. Correlation limits (no more than 2 same-direction trades)
4. Drawdown-based position reduction
5. Real-time P&L monitoring with circuit breakers
```

### **Priority 5: Market Regime Adaptation**

#### **Adaptive Parameters**
```python
# Market Regime Detection:
def adapt_to_market_regime(vix_level, market_trend, volatility):
    if vix_level < 15:  # Low vol environment
        return {
            'min_confidence': 0.45,     # Higher threshold
            'position_size_mult': 1.2,  # Larger positions
            'profit_target': 75%        # Lower targets
        }
    elif vix_level > 25:  # High vol environment  
        return {
            'min_confidence': 0.35,     # Lower threshold
            'position_size_mult': 0.8,  # Smaller positions
            'profit_target': 150%       # Higher targets
        }
```

## 🔬 Advanced Analytics & Monitoring

### **Real-Time Performance Tracking**
```python
# Live Metrics Dashboard:
- Win Rate (rolling 20 trades)
- Profit Factor (daily/weekly)
- Sharpe Ratio (risk-adjusted returns)
- Maximum Drawdown (real-time)
- Options Greeks exposure
- Market regime indicators
```

### **Strategy Health Checks**
```python
# Automated Monitoring:
1. Signal quality degradation detection
2. Market condition change alerts
3. Performance benchmark comparisons
4. Risk limit breach notifications
5. ThetaData connection health
```

## 🏗️ Development Roadmap

### **Phase 1: Risk Management (Week 1-2)**
- [ ] Implement dynamic exit strategies
- [ ] Add portfolio-level risk controls
- [ ] Optimize position sizing algorithms
- [ ] Add real-time P&L monitoring

### **Phase 2: Signal Quality (Week 3-4)**
- [ ] Enhance entry criteria with regime detection
- [ ] Add momentum confirmation filters
- [ ] Implement options flow analysis
- [ ] Optimize CALL vs PUT signal generation

### **Phase 3: Adaptive Systems (Week 5-6)**
- [ ] Market regime adaptation
- [ ] Machine learning signal validation
- [ ] Dynamic parameter optimization
- [ ] Performance benchmark integration

### **Phase 4: Production Ready (Week 7-8)**
- [ ] Live trading infrastructure
- [ ] Real-time monitoring dashboard
- [ ] Automated reporting systems
- [ ] Risk management circuit breakers

## 📞 Technical Support & Integration

### **ThetaData Requirements**
- ThetaData Terminal must be running
- Active subscription for SPY options data
- Stable internet connection for real-time feeds

### **Alpaca Requirements**  
- Alpaca account with market data access
- API keys configured in environment
- Paper trading recommended for initial testing

### **System Requirements**
- Python 3.8+
- 8GB RAM minimum
- SSD storage for data caching
- Low-latency internet connection

---

## 📈 Conclusion

This high-frequency 0DTE strategy successfully achieves the primary objective of **dramatically increasing trading frequency** from 0.18 to 8.2 trades per day (**45x improvement**). While current profitability needs optimization, the framework provides a solid foundation for:

1. ✅ **High-frequency signal generation**
2. ✅ **Smart option filtering** 
3. ✅ **Real ThetaData integration**
4. ✅ **Comprehensive risk management**
5. ✅ **Professional analytics framework**

The next phase focuses on **improving win rate from 34% to 45%+** and **optimizing exit strategies** to achieve consistent profitability while maintaining the high trading frequency that makes this strategy unique in the 0DTE options space.

---

*Strategy developed with Alpaca-py SDK, ThetaData, and Backtrader integration for professional options trading.* 