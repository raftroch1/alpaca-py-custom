# HISTORICAL DATA FOLDER ORGANIZATION

## Overview
This folder contains the historical data components of the alpaca-py SDK, organized into specialized subfolders for better maintainability and development workflow.

## Directory Structure
```
alpaca/data/historical/
├── strategies/              # All option trading strategies
│   ├── templates/          # Base template and examples
│   ├── logs/              # Strategy results and logs
│   ├── archive/           # Archived/deprecated strategies
│   └── [strategy_name]_v[X].py  # Versioned strategy files
├── backtests/              # Backtest implementations and analysis
├── thetadata/              # ThetaData infrastructure and testing
│   └── tests/             # ThetaData-specific tests
├── [SDK core files]        # Core historical data functionality
└── README.md              # This file
```

## Core SDK Files (Root Level)
These files remain in the root as they are part of the core alpaca-py SDK:
- `__init__.py` - Module initialization
- `corporate_actions.py` - Corporate actions data handling
- `crypto.py` - Cryptocurrency historical data client
- `news.py` - News data client
- `option.py` - Options historical data client  
- `screener.py` - Stock screening functionality
- `stock.py` - Stock historical data client
- `utils.py` - Utility functions for historical data

## Folder Purposes

### `/strategies/`
**Purpose**: All option trading strategy implementations
- Strategy development using standardized templates
- Versioned strategy files (v1, v2, etc.)
- Centralized logging and results
- Archive for deprecated strategies

### `/backtests/`
**Purpose**: Backtest implementations and analysis
- Performance testing frameworks
- Position sizing analysis
- Strategy optimization studies
- Historical performance validation

### `/thetadata/`
**Purpose**: ThetaData infrastructure and connectivity
- Client implementations for ThetaData API
- Connection testing and diagnostics
- Utilities for option data fetching
- ThetaTerminal.jar and configuration

## Development Workflow

### For New Strategies:
1. Inherit from `strategies/templates/base_theta_strategy.py`
2. Implement required abstract methods
3. Use proven ThetaData connection (no custom implementations)
4. Save results to `strategies/logs/`
5. Version your strategy files (`strategy_v1.py`, `strategy_v2.py`)

### For Backtests:
1. Create backtest files in `backtests/` folder
2. Use real ThetaData (no simulation)
3. Export results to `strategies/logs/`
4. Follow naming convention: `[strategy_name]_backtest.py`

### For ThetaData Work:
1. Use existing proven implementations in `thetadata/`
2. Test connectivity with diagnostic tools
3. Add new tests to `thetadata/tests/`
4. Follow established API format patterns

## Key Principles
- **Real Data Only**: No simulation fallbacks
- **Standardized Templates**: All strategies inherit from base template
- **Proper Versioning**: Strategy files use v1, v2, etc. naming
- **Centralized Logging**: All results go to `strategies/logs/`
- **Clean Organization**: Each folder has a specific purpose

## Quick Start
1. Review `strategies/STRATEGY_DEVELOPMENT_RULES.md` for development guidelines
2. Check `strategies/CURSORRULES_ADDITION.md` for .cursorrules content
3. Start with `strategies/templates/example_strategy_v1.py` as reference
4. Ensure ThetaData connectivity using `thetadata/thetadata_diagnosis.py`

This organization ensures maintainable, scalable, and standardized option strategy development within the alpaca-py ecosystem. 