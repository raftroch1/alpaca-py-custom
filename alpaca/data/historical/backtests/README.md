# BACKTESTS FOLDER

## Purpose
This folder contains all backtest implementations and analysis files for option trading strategies.

## Contents
- **Backtest Implementations**: Complete backtest scripts that test strategy performance
- **Analysis Files**: Position sizing analysis, performance analysis, and optimization studies
- **Historical Results**: Archived backtest results and performance metrics

## Current Files
- `corrected_thetadata_backtest.py` - Corrected ThetaData backtest implementation
- `corrected_position_sizing_backtest.py` - Position sizing analysis and backtesting
- `position_sizing_analysis.py` - Position sizing optimization studies

## Naming Convention
- Backtest files: `[strategy_name]_backtest.py`
- Analysis files: `[analysis_type]_analysis.py`
- Result files should be exported to `strategies/logs/` folder

## Usage
These backtests should inherit from the base strategy template in `strategies/templates/base_theta_strategy.py` for consistency and standardization.

## Note
All new backtests should use real ThetaData (no simulation) and follow the established framework patterns. 