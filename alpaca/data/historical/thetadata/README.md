# THETADATA FOLDER

## Purpose
This folder contains all ThetaData-related infrastructure, clients, utilities, and testing files.

## Contents
- **Client Implementations**: Various ThetaData client implementations
- **Utilities**: Helper functions and data processing tools
- **Testing**: Comprehensive test suite for ThetaData connectivity and functionality
- **Configuration**: ThetaTerminal.jar and related setup files

## Current Files

### Core Clients
- `client.py` - Main ThetaData client implementation
- `thetadata_rest_client.py` - REST API client for ThetaData
- `thetadata_socket_client.py` - WebSocket client for real-time data

### Utilities & Tools
- `utils.py` - Utility functions for ThetaData operations
- `option_chain_fetcher.py` - Option chain fetching utilities
- `thetadata_diagnosis.py` - Diagnostic tools for ThetaData connectivity
- `ThetaTerminal.jar` - ThetaData Terminal application

### Tests Directory (`tests/`)
- `connectivity_tests.py` - Basic connectivity testing
- `test_thetadata_diagnosis.py` - Diagnostic testing suite
- `test_connection.py` - Connection testing utilities
- `thetadata_test.py` - General ThetaData testing
- `test_spy_option_20250505.py` - Specific option data testing

## Proven Working Configuration
The current implementation uses direct HTTP calls to `http://127.0.0.1:25510/v2/hist/option/eod` with:
- Strike prices in thousandths (multiply by 1000)
- Dates in YYYYMMDD format
- Response parsing: close price at index 5

## Usage Notes
- **ALWAYS** use the proven working ThetaData format from `base_theta_strategy.py`
- NO custom ThetaData implementations - use the standardized approach
- All new strategies should inherit from the base template
- ThetaTerminal.jar must be running for local data access

## Connection Requirements
1. ThetaData subscription and credentials
2. ThetaTerminal.jar running locally on port 25510
3. Proper API key configuration
4. Network connectivity to ThetaData servers

## Troubleshooting
Use `thetadata_diagnosis.py` for connection testing and diagnostics. 