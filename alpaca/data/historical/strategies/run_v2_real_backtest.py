#!/usr/bin/env python3
"""
Run V2-REAL Strategy Backtest with Real ThetaData
"""

from high_frequency_0dte_v2_real import TrueHighFrequency0DTEStrategyV2Real
from datetime import datetime, timedelta

def main():
    print("🚀 Starting V2-REAL Backtest with Real ThetaData")
    
    # Initialize strategy
    strategy = TrueHighFrequency0DTEStrategyV2Real()
    
    # Set date range (recent 5 trading days) in YYYYMMDD format
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
    
    print(f"📅 Backtesting from {start_date} to {end_date}")
    
    # Run backtest
    results = strategy.run_backtest(start_date, end_date)
    
    print("✅ Backtest completed!")
    return results

if __name__ == "__main__":
    main()
