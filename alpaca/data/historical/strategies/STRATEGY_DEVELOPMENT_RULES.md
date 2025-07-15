# STRATEGY DEVELOPMENT FRAMEWORK RULES

## Overview
This document defines the standardized process for developing, testing, and maintaining option trading strategies in the alpaca-py project using ThetaData.

## Directory Structure
```
alpaca/data/historical/
├── strategies/
│   ├── templates/
│   │   ├── base_theta_strategy.py      # Base template class
│   │   └── example_strategy_v1.py      # Example implementation
│   ├── logs/                           # All log files and CSV results
│   ├── archive/                        # Failed/old backtest files
│   ├── [strategy_name]_v1.py          # Working strategy version 1
│   ├── [strategy_name]_v2.py          # Improved strategy version 2
│   └── ...
├── [core SDK files remain in root]
```

## Strategy Development Process

### 1. Strategy Creation
- **ALWAYS** inherit from `BaseThetaStrategy` template
- Use proven ThetaData connection and API format
- Implement required abstract methods:
  - `analyze_market_conditions()`
  - `execute_strategy()`
  - `calculate_position_size()`

### 2. Naming Convention
- **Base strategies**: `[strategy_name]_v1.py`
- **Improved versions**: `[strategy_name]_v2.py`, `[strategy_name]_v3.py`, etc.
- **Strategy name examples**: `vix_contrarian`, `iron_condor`, `delta_neutral`
- **Class name**: PascalCase (e.g., `VixContrarianStrategy`)

### 3. Versioning Rules
- Keep ALL working versions for comparison
- When improving a strategy, create new version file
- Include version comparison in commit messages
- Document changes between versions in docstring

### 4. Logging Requirements
- **Automatic logging** to `strategies/logs/` folder
- **Filename format**: `{strategy_name}_{version}_{timestamp}.log`
- **CSV results**: `{strategy_name}_{version}_{timestamp}_trades.csv`
- **Log both**: file and console output

### 5. Data Requirements
- **MUST use real ThetaData only** - NO simulation fallback
- **Proper error handling** for missing option data
- **Skip trades** when real data unavailable (don't simulate)
- **Validate ThetaData connection** before strategy execution

### 6. File Organization
- **Clean historical folder** of failed backtests before new development
- **Archive old files** to `strategies/archive/`
- **Move CSV results** to `strategies/logs/`
- **Keep core SDK files** in root historical folder

## Code Template Usage

### Base Template Import
```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'strategies', 'templates'))
from base_theta_strategy import BaseThetaStrategy
```

### Strategy Implementation Template
```python
class YourStrategy(BaseThetaStrategy):
    def __init__(self):
        super().__init__(
            strategy_name="your_strategy",
            version="v1",
            starting_capital=25000,
            max_risk_per_trade=0.03,
            target_profit_per_trade=0.002
        )
    
    def analyze_market_conditions(self, spy_price: float, vix_level: float, date: str):
        # Your market analysis logic
        return {
            'should_trade': True/False,
            'strategy_type': 'YOUR_STRATEGY_TYPE',
            # Other analysis results
        }
    
    def execute_strategy(self, market_analysis, spy_price: float, date: str):
        # Your strategy execution logic
        # Use self.get_option_price() for real ThetaData
        # Return trade dictionary or None if skipped
        
    def calculate_position_size(self, strategy_type: str, premium: float):
        # Your position sizing logic
        # Return number of contracts
```

## ThetaData Best Practices

### Proven API Format
- **Endpoint**: `http://127.0.0.1:25510/v2/hist/option/eod`
- **Strike format**: Multiply by 1000 (e.g., $535 → 535000)
- **Date format**: YYYYMMDD
- **Response parsing**: Close price at index [5]

### Error Handling
- **Always check** for valid response status (200)
- **Validate data exists** in response['response']
- **Skip trades** when option data unavailable
- **Log warnings** for missing data, not errors

### Option Price Retrieval
```python
# Use base template method
option_price = self.get_option_price(
    symbol='SPY',
    exp_date='20250131',  # YYYYMMDD
    strike=600.0,         # Dollar amount
    right='C',           # 'C' or 'P'
    date='20250131'      # Trade date YYYYMMDD
)
```

## Performance Tracking

### Required Metrics
- Total return percentage
- Total profit/loss
- Win rate
- Average profit per trade
- Number of trades executed
- Number of trades skipped

### Logging Standards
- **Real-time logging** during strategy execution
- **Performance summary** at end of backtest
- **CSV export** of all trade details
- **Timestamp all files** for version tracking

## Quality Standards

### Code Quality
- **Type hints** on all methods
- **Comprehensive docstrings**
- **Error handling** for all external API calls
- **Unit tests** for position sizing logic

### Testing Requirements
- **ThetaData connection test** before strategy run
- **Validation** of option pricing logic
- **Edge case handling** for extreme market conditions
- **Performance regression testing** between versions

## Deployment Checklist

### Before Committing
- [ ] Strategy inherits from BaseThetaStrategy
- [ ] All abstract methods implemented
- [ ] ThetaData connection tested
- [ ] Logging working correctly
- [ ] CSV results saved to logs folder
- [ ] No simulation fallback used
- [ ] Historical folder cleaned
- [ ] Version number incremented

### File Cleanup
- [ ] Archive failed backtest files
- [ ] Move CSV results to logs
- [ ] Remove debug files
- [ ] Clean __pycache__ folders
- [ ] Remove temporary images/plots

## Example Workflow

1. **Start new strategy**:
   ```bash
   cd alpaca/data/historical/strategies
   cp templates/example_strategy_v1.py my_strategy_v1.py
   ```

2. **Implement strategy logic**:
   - Modify class name and strategy_name
   - Implement the three abstract methods
   - Test with small date range first

3. **Clean and organize**:
   ```bash
   # Archive old files
   mv ../failed_backtest*.py archive/
   mv ../*.csv logs/
   
   # Run strategy
   python my_strategy_v1.py
   ```

4. **Version improvements**:
   ```bash
   # Create new version
   cp my_strategy_v1.py my_strategy_v2.py
   # Implement improvements
   # Update version="v2" in __init__
   ```

## Troubleshooting

### Common Issues
- **ThetaData connection fails**: Check ThetaTerminal is running
- **No option data**: Verify date format (YYYYMMDD) and strike (thousandths)
- **Linter errors**: Ensure proper imports and type hints
- **Empty results**: Check if trades being skipped due to missing data

### Debug Process
1. Test ThetaData connection manually
2. Check single option price retrieval
3. Verify date formatting
4. Review logging output for skipped trades
5. Validate position sizing calculations

## Support Files
- `base_theta_strategy.py`: Main template class
- `example_strategy_v1.py`: Working example implementation
- `thetadata_diagnosis.py`: Connection testing utility
- This document: Development guidelines and best practices 