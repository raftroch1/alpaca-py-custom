#!/usr/bin/env python3
"""
V2-REAL BACKTEST RUNNER
Run the True High Frequency 0DTE Strategy V2-REAL with real ThetaData

This script imports and executes the V2-REAL strategy for backtesting.
"""

import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from high_frequency_0dte_v2_real import TrueHighFrequency0DTEStrategyV2Real
    print("✅ Successfully imported TrueHighFrequency0DTEStrategyV2Real")
except ImportError as e:
    print(f"❌ Failed to import strategy: {e}")
    sys.exit(1)

def main():
    """Run the V2-REAL backtest"""
    print("="*80)
    print("🚀 STARTING TRUE HIGH FREQUENCY 0DTE STRATEGY V2-REAL BACKTEST")
    print("📊 Using REAL ThetaData - NO SIMULATION")
    print("="*80)
    
    try:
        # Initialize strategy
        strategy = TrueHighFrequency0DTEStrategyV2Real()
        print("✅ V2-REAL Strategy initialized successfully")
        
        # Set date range (recent trading days)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=10)  # Look back 10 days for 5 trading days
        
        print(f"📅 Backtesting from {start_date} to {end_date}")
        print("📊 USING REAL THETADATA - NO SIMULATION")
        
        # Run backtest
        results = strategy.run_backtest(
            start_date=start_date,
            end_date=end_date,
            use_real_data=True  # Force real data
        )
        
        print("✅ Backtest completed successfully")
        return results
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
    if results:
        print("🎉 Backtest completed with results")
    else:
        print("❌ Backtest failed") 