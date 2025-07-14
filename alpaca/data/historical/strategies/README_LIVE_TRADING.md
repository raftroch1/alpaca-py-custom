# Live High Conviction 0DTE Options Trading

This directory contains the live paper trading implementation of our High Conviction 0DTE options strategy using Alpaca's API.

## ⚠️ IMPORTANT SAFETY NOTICE

- **PAPER TRADING ONLY**: This implementation uses Alpaca's paper trading environment
- **No Real Money**: All trades are simulated with virtual money
- **Educational Purpose**: Use this to validate strategy performance before considering live trading
- **Monitor Actively**: Always supervise automated trading systems

## Strategy Overview

The High Conviction strategy achieved **538.15% returns** in backtesting by:

1. **Selective Trading**: Only trades when conviction score ≥ 5
2. **Diagonal Spreads**: Focuses on proven profitable strategy 
3. **Large Position Sizing**: 150-250 contracts when conditions are perfect
4. **Risk Management**: Automatic stop losses and profit taking
5. **Quality over Quantity**: Maximum 3 trades per day

### Key Performance Metrics (Backtest)
- **Average P&L per Trade**: $3,449.67
- **Win Rate**: 69.2%
- **Big Wins (>$500)**: 69.2% of trades
- **Best Day**: $7,606.47
- **Strategy**: 100% diagonal spreads

## Files

- **`live_high_conviction_trader.py`** - Main live trading bot
- **`setup_live_trader.py`** - Setup and validation script
- **`requirements_live_trader.txt`** - Required Python packages
- **`high_conviction_zero_dte_strategy.py`** - Backtested strategy (reference)

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements_live_trader.txt
```

### 2. Get Alpaca Paper Trading Account

1. Go to [https://alpaca.markets/](https://alpaca.markets/)
2. Sign up for a free account
3. Navigate to "Paper Trading" section
4. Generate API Key and Secret Key

### 3. Set Environment Variables

**Option A: Environment Variables (Recommended)**
```bash
export ALPACA_API_KEY="your_paper_api_key_here"
export ALPACA_SECRET_KEY="your_paper_secret_key_here"
```

**Option B: Set during setup**
The setup script will prompt you for keys if not found in environment.

### 4. Run Setup Validation

```bash
python setup_live_trader.py
```

This will:
- Check all dependencies
- Validate API keys
- Test account connection
- Verify options trading permissions
- Show usage instructions

### 5. Start Live Trading

```bash
python live_high_conviction_trader.py
```

## How It Works

### Market Analysis
- **VIX Monitoring**: Real-time VIX data from Yahoo Finance
- **SPY Price Tracking**: Current SPY price from Alpaca
- **Conviction Scoring**: Multi-factor analysis (VIX range, stability, momentum, timing)

### High Conviction Criteria
- VIX in optimal range (10.5-16.0)
- VIX stability or declining trend
- Controlled SPY momentum
- Favorable day of week (Tue-Thu)

### Trade Execution
- **Strategy**: Diagonal spreads with SPY 0DTE PUT options
- **Position Sizing**: 150-250 contracts based on conviction
- **Risk Management**: Max $1,500 risk per trade
- **Entry**: Market orders for immediate execution

### Position Management
- **Monitoring**: Real-time P&L tracking
- **Exit Conditions**:
  - Profit target: $20+ per contract
  - Stop loss: -$8 per contract
  - Time exit: 3:45 PM ET (market close)

### Daily Limits
- Maximum 3 trades per day
- Skip trading if no high conviction setups
- Comprehensive daily performance logging

## Market Hours

- **Trading Window**: 9:30 AM - 4:00 PM ET
- **Days**: Monday - Friday
- **Position Closing**: Automatic before 4:00 PM ET

## Logging and Monitoring

All trading activity is logged to:
- **Console**: Real-time updates
- **File**: `live_trading.log`

### Log Information
- Market data updates
- Conviction analysis results
- Trade executions
- Position monitoring
- Daily performance summaries

## Risk Management Features

### Position Level
- **Stop Loss**: -$8 per contract
- **Profit Target**: $20+ per contract
- **Time-based Exit**: Before market close

### Account Level
- **Daily Risk Limit**: Max 3 trades
- **Position Size Limits**: Based on account value
- **Maximum Risk**: 6% of account per trade

### Strategy Level
- **Conviction Filtering**: Only high probability setups
- **Market Condition Analysis**: Multi-factor validation
- **Automated Monitoring**: Continuous position oversight

## Performance Tracking

The bot tracks:
- Daily P&L and returns
- Trade win rate and average profit
- Conviction score effectiveness
- Account value progression

### Daily Summary Example
```
==========================================
DAILY TRADING SUMMARY - 2024-01-15
==========================================
Starting Account Value: $25,000.00
Current Account Value: $27,450.00
Daily P&L: $2,450.00
Daily Return: 9.80%
Trades Executed: 2
Active Positions: 0
==========================================
```

## Options Trading Requirements

### Alpaca Account Requirements
- **Options Trading Level**: 2+ (Level 3+ recommended for multi-leg)
- **Account Type**: Paper Trading
- **Minimum Equity**: $2,000 (recommended $25,000+)

### Strategy Requirements
- **SPY Options**: Must be enabled for options trading
- **0DTE Options**: Same-day expiration options
- **Multi-leg Orders**: For diagonal spreads

## Troubleshooting

### Common Issues

**"Options trading level insufficient"**
- Apply for higher options trading level in Alpaca dashboard
- Level 3+ required for multi-leg strategies

**"No high conviction setups"**
- Normal behavior - strategy is selective
- Requires specific VIX and market conditions
- May skip entire days if conditions aren't optimal

**"VIX data unavailable"**
- Check internet connection
- Yahoo Finance API may have temporary issues
- Fallback values are used automatically

**"Options contracts not found"**
- Ensure market is open
- SPY 0DTE options may not be available on some days
- Check if it's a holiday or early market close

### Support

For issues with:
- **Alpaca API**: Check [Alpaca Documentation](https://docs.alpaca.markets/)
- **Strategy Logic**: Review backtest results in other files
- **Setup Problems**: Re-run `setup_live_trader.py`

## Performance Expectations

Based on backtesting:
- **Not every day will be profitable**
- **High conviction days can generate $1,000-7,000+ profits**
- **Strategy is designed for quality over quantity**
- **Expect 1-3 trades per week typically**

## Next Steps

1. **Run for 1 week** to validate live performance
2. **Compare results** to backtest expectations  
3. **Analyze logs** for improvement opportunities
4. **Adjust parameters** if needed based on live results

## Disclaimer

This is educational software for paper trading only. Past performance does not guarantee future results. Options trading involves substantial risk and is not suitable for all investors. Always understand the risks before trading. 