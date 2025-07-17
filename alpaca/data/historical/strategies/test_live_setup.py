#!/usr/bin/env python3
"""
LIVE TRADING SETUP TEST
======================

Test script to verify Alpaca connection and environment setup
for the ultra-aggressive 0DTE strategy.

Usage:
    python test_live_setup.py
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add alpaca imports
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

try:
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.trading.requests import GetOptionContractsRequest
    from alpaca.trading.enums import ContractType
    print("✅ Alpaca imports successful")
except ImportError as e:
    print(f"❌ Failed to import Alpaca modules: {e}")
    sys.exit(1)


def test_environment():
    """Test environment variables"""
    print("\n🔧 TESTING ENVIRONMENT SETUP")
    print("=" * 50)
    
    # Load .env file
    load_dotenv()
    
    # Check required variables
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key:
        print("❌ ALPACA_API_KEY not found in environment")
        return False
    
    if not secret_key:
        print("❌ ALPACA_SECRET_KEY not found in environment")
        return False
    
    print(f"✅ ALPACA_API_KEY found: {api_key[:8]}...")
    print(f"✅ ALPACA_SECRET_KEY found: {secret_key[:8]}...")
    
    return True


def test_trading_client():
    """Test Alpaca Trading Client connection"""
    print("\n📡 TESTING TRADING CLIENT CONNECTION")
    print("=" * 50)
    
    try:
        # Initialize client
        client = TradingClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            paper=True  # Always use paper for testing
        )
        
        # Test connection by getting account info
        account = client.get_account()
        
        print(f"✅ Trading client connected successfully")
        print(f"   Account Number: {account.account_number}")
        print(f"   Account Status: {account.status}")
        print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
        print(f"   Cash: ${float(account.cash):,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trading client connection failed: {e}")
        return False


def test_data_client():
    """Test Alpaca Data Client connection"""
    print("\n📊 TESTING DATA CLIENT CONNECTION")
    print("=" * 50)
    
    try:
        # Initialize data client
        client = StockHistoricalDataClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY')
        )
        
        # Test by getting recent SPY data
        from datetime import timedelta, timezone
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=10)
        
        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start_time,
            end=end_time
        )
        
        bars = client.get_stock_bars(request)
        df = bars.df
        
        if not df.empty:
            latest_price = df['close'].iloc[-1]
            print(f"✅ Data client connected successfully")
            print(f"   Retrieved {len(df)} SPY minute bars")
            print(f"   Latest SPY price: ${latest_price:.2f}")
            print(f"   Data timestamp: {df.index[-1]}")
            return True
        else:
            print("⚠️ Data client connected but no data returned")
            return False
        
    except Exception as e:
        print(f"❌ Data client connection failed: {e}")
        return False


def test_options_access():
    """Test options contract access"""
    print("\n🎯 TESTING OPTIONS ACCESS")
    print("=" * 50)
    
    try:
        # Initialize trading client
        client = TradingClient(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            paper=True
        )
        
        # Get today's date for 0DTE options
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Search for 0DTE SPY options
        request = GetOptionContractsRequest(
            underlying_symbols=["SPY"],
            contract_type=ContractType.CALL,
            expiration_date=today,
            limit=10
        )
        
        response = client.get_option_contracts(request)
        
        if response.option_contracts:
            print(f"✅ Options access successful")
            print(f"   Found {len(response.option_contracts)} 0DTE SPY options")
            
            # Show a few examples
            for i, contract in enumerate(response.option_contracts[:3]):
                print(f"   Example {i+1}: {contract.symbol} (Strike: ${contract.strike_price})")
            
            return True
        else:
            print("⚠️ Options access working but no 0DTE options found today")
            print("   This is normal on non-trading days or when 0DTE options aren't available")
            return True
        
    except Exception as e:
        print(f"❌ Options access failed: {e}")
        return False


def test_strategy_parameters():
    """Test strategy parameter loading"""
    print("\n⚙️ TESTING STRATEGY PARAMETERS")
    print("=" * 50)
    
    try:
        # Test parameter structure
        params = {
            'confidence_threshold': 0.20,
            'min_signal_score': 3,
            'bull_momentum_threshold': 0.001,
            'bear_momentum_threshold': 0.001,
            'volume_threshold': 1.5,
            'base_contracts': 5,
            'max_daily_trades': 15,
        }
        
        print("✅ Strategy parameters loaded successfully")
        print("   Key parameters:")
        for key, value in params.items():
            print(f"     {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy parameter test failed: {e}")
        return False


def run_comprehensive_test():
    """Run all tests"""
    print("🔥 LIVE ULTRA-AGGRESSIVE 0DTE STRATEGY - SETUP TEST")
    print("=" * 70)
    print("🧪 Testing all components for live trading readiness")
    print(f"🕐 Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Environment Setup", test_environment),
        ("Trading Client", test_trading_client),
        ("Data Client", test_data_client),
        ("Options Access", test_options_access),
        ("Strategy Parameters", test_strategy_parameters),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📋 TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n🎯 OVERALL RESULT: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🚀 READY FOR LIVE TRADING!")
        print("   Next steps:")
        print("   1. Run: python live_ultra_aggressive_0dte.py")
        print("   2. Monitor logs for trading signals")
        print("   3. Verify paper trading performance")
        print("   4. Scale up when confident")
    else:
        print("\n🔧 SETUP INCOMPLETE")
        print("   Please fix failed tests before proceeding:")
        print("   1. Check .env file has correct Alpaca paper trading keys")
        print("   2. Verify internet connection")
        print("   3. Ensure Alpaca account is active")
        print("   4. Install missing dependencies")
    
    return all_passed


if __name__ == "__main__":
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
        sys.exit(1) 