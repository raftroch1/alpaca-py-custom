# ADD THIS TO YOUR .CURSORRULES FILE

## Option Strategy Development Framework

### ThetaData Strategy Development Process

When developing option strategies in the alpaca-py project, you must follow this standardized framework:

#### 1. Strategy Template Usage
- **ALWAYS** inherit from `BaseThetaStrategy` class in `alpaca/data/historical/strategies/templates/base_theta_strategy.py`
- Use proven ThetaData connection and API format - NO custom implementations
- Implement the three required abstract methods:
  - `analyze_market_conditions(spy_price, vix_level, date)` → Dict[str, Any]
  - `execute_strategy(market_analysis, spy_price, date)` → Optional[Dict[str, Any]]
  - `calculate_position_size(strategy_type, premium)` → int

#### 2. File Organization and Naming
- **Strategy files**: `alpaca/data/historical/strategies/[strategy_name]_v[number].py`
- **Versioning**: Keep ALL versions (v1, v2, v3, etc.) for comparison
- **Logging**: Automatic to `strategies/logs/` with timestamp
- **Results**: CSV files auto-saved to `strategies/logs/`
- **Archive**: Move failed backtests to `strategies/archive/`

#### 3. ThetaData Requirements
- **MUST use real ThetaData only** - absolutely NO simulation fallback
- **Proven API format**: 
  - Endpoint: `http://127.0.0.1:25510/v2/hist/option/eod`
  - Strike format: multiply by 1000 (e.g., $535 → 535000)
  - Date format: YYYYMMDD
  - Response parsing: close price at index [5]
- **Error handling**: Skip trades when real data unavailable (don't simulate)
- **Validation**: Test ThetaData connection before strategy execution

#### 4. Code Quality Standards
- **Type hints** on all methods and parameters
- **Comprehensive logging** with emoji indicators (🚀 ✅ ❌ ⚠️ 📊 💰)
- **Error handling** for all external API calls
- **Performance metrics** calculation and reporting
- **CSV export** of all trade details with timestamps

#### 5. Pre-Development Cleanup
Before starting new strategy development:
```bash
# Archive failed backtests
mv alpaca/data/historical/*failed*.py alpaca/data/historical/strategies/archive/
mv alpaca/data/historical/*zero_dte*.py alpaca/data/historical/strategies/archive/
mv alpaca/data/historical/*real_data*.py alpaca/data/historical/strategies/archive/

# Move CSV results to logs
mv alpaca/data/historical/*.csv alpaca/data/historical/strategies/logs/

# Clean temporary files
rm -rf alpaca/data/historical/__pycache__
rm -f alpaca/data/historical/*.png
```

#### 6. Strategy Development Template
```python
#!/usr/bin/env python3
"""
[STRATEGY_NAME] V[NUMBER]
Brief description of strategy approach and market conditions.

Usage:
    python [strategy_name]_v[number].py
"""

import sys
import os
from typing import Dict, Any, Optional

# Import base template
sys.path.append(os.path.join(os.path.dirname(__file__), 'templates'))
from base_theta_strategy import BaseThetaStrategy

class YourStrategy(BaseThetaStrategy):
    def __init__(self):
        super().__init__(
            strategy_name="your_strategy",
            version="v1",
            starting_capital=25000,
            max_risk_per_trade=0.03,
            target_profit_per_trade=0.002
        )
    
    def analyze_market_conditions(self, spy_price: float, vix_level: float, date: str) -> Dict[str, Any]:
        # Market analysis logic here
        pass
    
    def execute_strategy(self, market_analysis: Dict[str, Any], spy_price: float, date: str) -> Optional[Dict[str, Any]]:
        # Strategy execution logic here
        pass
    
    def calculate_position_size(self, strategy_type: str, premium: float) -> int:
        # Position sizing logic here
        pass

if __name__ == "__main__":
    strategy = YourStrategy()
    strategy.run_backtest('2025-01-01', '2025-06-30')
```

#### 7. Version Management
- **New strategy**: Start with `_v1.py`
- **Improvements**: Create `_v2.py`, `_v3.py`, etc.
- **Never overwrite** working versions
- **Document changes** in version docstrings
- **Performance comparison** between versions in commit messages

#### 8. Deployment Checklist
Before committing any strategy:
- [ ] Inherits from BaseThetaStrategy
- [ ] All abstract methods implemented
- [ ] ThetaData connection tested successfully
- [ ] No simulation fallback code present
- [ ] Logging working to strategies/logs/
- [ ] CSV results auto-exported
- [ ] Historical folder cleaned of failed backtests
- [ ] Version number properly incremented

#### 9. Debugging Process
If strategy fails:
1. Test ThetaData connection manually
2. Verify date formatting (YYYYMMDD)
3. Check strike price format (thousandths)
4. Review log files for missing data warnings
5. Validate position sizing calculations
6. Use `thetadata_diagnosis.py` for connection testing

#### 10. Performance Standards
All strategies must report:
- Total return percentage
- Win rate
- Average profit per trade
- Number of trades executed vs. skipped
- Real data usage percentage (must be 100%)

### Reference Files
- **Base template**: `alpaca/data/historical/strategies/templates/base_theta_strategy.py`
- **Example**: `alpaca/data/historical/strategies/templates/example_strategy_v1.py`
- **Working strategy**: `alpaca/data/historical/strategies/vix_contrarian_v1.py`
- **Documentation**: `alpaca/data/historical/strategies/STRATEGY_DEVELOPMENT_RULES.md`

### Important Notes
- **Never** create strategies that don't inherit from BaseThetaStrategy
- **Never** implement custom ThetaData connections (use proven template)
- **Never** include simulation fallback logic
- **Always** clean historical folder before new development
- **Always** version strategies incrementally
- **Always** log comprehensive performance metrics 