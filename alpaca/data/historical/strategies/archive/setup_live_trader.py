#!/usr/bin/env python3
"""
Setup script for Live High Conviction Trader

This script helps you configure and validate your setup before running live trades.
"""

import os
import sys
from alpaca.trading.client import TradingClient

def check_dependencies():
    """Check if all required dependencies are installed."""
    
    print("Checking dependencies...")
    
    required_packages = [
        'alpaca',
        'yfinance', 
        'numpy',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements_live_trader.txt")
        return False
    
    print("All dependencies installed!")
    return True

def setup_api_keys():
    """Guide user through API key setup."""
    
    print("\n" + "="*60)
    print("ALPACA API KEY SETUP")
    print("="*60)
    
    # Check if already set
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if api_key and secret_key:
        print("✓ API keys found in environment variables")
        return api_key, secret_key
    
    print("API keys not found in environment variables.")
    print("\nTo get your Alpaca paper trading API keys:")
    print("1. Go to https://alpaca.markets/")
    print("2. Sign up for a free account")
    print("3. Go to 'Paper Trading' section")
    print("4. Generate API Key and Secret")
    print("5. Set environment variables:")
    print("   export ALPACA_API_KEY='your_key_here'")
    print("   export ALPACA_SECRET_KEY='your_secret_here'")
    
    # Option to enter keys for this session
    print("\nOr enter keys for this session only:")
    api_key = input("Enter your Alpaca API Key: ").strip()
    secret_key = input("Enter your Alpaca Secret Key: ").strip()
    
    if not api_key or not secret_key:
        print("Error: Both API key and secret are required")
        return None, None
    
    # Set for this session
    os.environ['ALPACA_API_KEY'] = api_key
    os.environ['ALPACA_SECRET_KEY'] = secret_key
    
    return api_key, secret_key

def validate_account(api_key, secret_key):
    """Validate Alpaca account and check options trading permissions."""
    
    print("\n" + "="*60)
    print("ACCOUNT VALIDATION")
    print("="*60)
    
    try:
        # Test connection
        trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True
        )
        
        account = trading_client.get_account()
        
        print(f"✓ Connection successful!")
        print(f"✓ Account ID: {getattr(account, 'id', 'Unknown')}")
        print(f"✓ Account Status: {getattr(account, 'status', 'Unknown')}")
        
        # Safely get account values
        equity = getattr(account, 'equity', '0')
        buying_power = getattr(account, 'buying_power', '0')
        
        try:
            equity_val = float(equity) if equity else 0.0
            bp_val = float(buying_power) if buying_power else 0.0
            print(f"✓ Equity: ${equity_val:,.2f}")
            print(f"✓ Buying Power: ${bp_val:,.2f}")
        except (ValueError, TypeError):
            print(f"✓ Equity: {equity}")
            print(f"✓ Buying Power: {buying_power}")
        
        # Check options trading
        options_level = getattr(account, 'options_trading_level', 0)
        options_bp = getattr(account, 'options_buying_power', 0)
        
        print(f"✓ Options Trading Level: {options_level}")
        
        try:
            options_bp_val = float(options_bp) if options_bp else 0.0
            print(f"✓ Options Buying Power: ${options_bp_val:,.2f}")
        except (ValueError, TypeError):
            print(f"✓ Options Buying Power: {options_bp}")
        
        if options_level < 2:
            print("\n⚠️  WARNING: Options trading level is less than 2.")
            print("   You may need to apply for options trading permissions.")
            print("   Multi-leg strategies require level 3+")
        else:
            print("✓ Options trading enabled!")
        
        return True
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nPossible issues:")
        print("- Incorrect API keys")
        print("- API keys are for live trading (we need paper trading)")
        print("- Network connectivity issues")
        return False

def show_usage_instructions():
    """Show instructions for running the live trader."""
    
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS")
    print("="*60)
    
    print("\nTo run the live trader:")
    print("1. Ensure market is open (9:30 AM - 4:00 PM ET, weekdays)")
    print("2. Run: python live_high_conviction_trader.py")
    print("\nStrategy Overview:")
    print("- Only trades on high conviction setups (score ≥ 5)")
    print("- Focuses on diagonal spreads with SPY 0DTE options")
    print("- Maximum 3 trades per day")
    print("- Automatic position monitoring and exit management")
    print("- Risk management: max $1,500 per trade")
    
    print("\nSafety Notes:")
    print("- This uses PAPER TRADING only")
    print("- Monitor your positions actively")
    print("- Keep trading logs for analysis")
    print("- Start with small position sizes to validate strategy")

def main():
    """Main setup function."""
    
    print("High Conviction Live Trader Setup")
    print("="*40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup API keys
    api_key, secret_key = setup_api_keys()
    if not api_key or not secret_key:
        sys.exit(1)
    
    # Validate account
    if not validate_account(api_key, secret_key):
        sys.exit(1)
    
    # Show usage instructions
    show_usage_instructions()
    
    print("\n✓ Setup complete! You're ready to run the live trader.")
    print("\nRun: python live_high_conviction_trader.py")

if __name__ == "__main__":
    main() 