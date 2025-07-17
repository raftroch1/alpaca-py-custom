"""
Diagonal Spread Strategy Backtest

Backtest file for diagonal_spread_strategy.py
Follows the naming convention: strategy_name.py -> strategy_name_backtest.py
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add strategies directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'strategies'))

try:
    from diagonal_spread_strategy import DiagonalSpreadStrategy
except ImportError as e:
    print(f"Could not import DiagonalSpreadStrategy: {e}")
    print("Make sure diagonal_spread_strategy.py exists in the strategies/ directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/diagonal_spread_backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DiagonalSpreadBacktest:
    """
    Backtest framework for Diagonal Spread Strategy
    """
    
    def __init__(self, 
                 start_date: str = "2024-01-01",
                 end_date: str = "2024-06-30",
                 initial_capital: float = 25000):
        
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        
        # Initialize strategy
        self.strategy = DiagonalSpreadStrategy(
            starting_capital=initial_capital
        )
        
        # Results storage
        self.backtest_results = {}
        self.trade_log = []
        
        logger.info(f"Initialized Diagonal Spread Backtest: {start_date} to {end_date}")

    def run_backtest(self) -> dict:
        """
        Run the diagonal spread strategy backtest
        """
        logger.info("🚀 Starting Diagonal Spread Strategy Backtest")
        
        try:
            # TODO: Implement actual backtesting logic
            # This is a template - needs to be completed based on the strategy implementation
            
            # Placeholder results
            self.backtest_results = {
                'strategy_name': 'diagonal_spread_strategy',
                'start_date': self.start_date,
                'end_date': self.end_date,
                'initial_capital': self.initial_capital,
                'status': 'template_created',
                'message': 'Backtest template created. Needs implementation based on strategy logic.'
            }
            
            logger.info("Backtest template completed")
            return self.backtest_results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {'error': str(e)}

    def generate_report(self):
        """
        Generate backtest performance report
        """
        print("\n" + "="*60)
        print("DIAGONAL SPREAD STRATEGY BACKTEST REPORT")
        print("="*60)
        print(f"Strategy: {self.backtest_results.get('strategy_name', 'N/A')}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Status: {self.backtest_results.get('status', 'N/A')}")
        print(f"Message: {self.backtest_results.get('message', 'N/A')}")
        print("="*60)

def main():
    """
    Run the diagonal spread backtest
    """
    print("🚀 Diagonal Spread Strategy Backtest")
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Initialize and run backtest
    backtest = DiagonalSpreadBacktest(
        start_date="2024-01-01",
        end_date="2024-06-30",
        initial_capital=25000
    )
    
    # Run backtest
    results = backtest.run_backtest()
    
    # Generate report
    backtest.generate_report()
    
    return results

if __name__ == "__main__":
    main() 