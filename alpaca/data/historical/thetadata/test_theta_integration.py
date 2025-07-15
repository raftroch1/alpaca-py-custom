#!/usr/bin/env python3
"""
TEST THETADATA BACKTRADER INTEGRATION
Quick test to verify the new integration is working properly
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add the project root to sys.path for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add the path to import our new integration
sys.path.append(os.path.dirname(__file__))

# Patch: Add backtrader_backtesting folder to sys.path for imports
backtrader_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backtests', 'backtrader_backtesting'))
if backtrader_path not in sys.path:
    sys.path.insert(0, backtrader_path)

def test_theta_connection():
    """Test ThetaData connection"""
    print("🔍 Testing ThetaData Connection...")
    
    try:
        from theta_multi_regime_backtrader import ThetaDataConnector
        
        connector = ThetaDataConnector()
        is_connected = connector.test_connection()
        
        if is_connected:
            print("✅ ThetaData connection successful")
            
            # Test option price fetch
            test_date = '2024-07-05'
            test_price = connector.get_option_price("SPY", test_date, 535.0, "P")
            
            if test_price is not None:
                print(f"✅ Option price fetch successful: SPY 535P @ ${test_price:.2f}")
            else:
                print("⚠️  Option price fetch returned None (may be expected for test date)")
                
        else:
            print("❌ ThetaData connection failed")
            print("   Make sure ThetaData Terminal is running")
            print("   URL: http://127.0.0.1:25510")
        
        return is_connected
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing ThetaData: {e}")
        return False

def test_strategy_components():
    """Test strategy components"""
    print("\n🔍 Testing Strategy Components...")
    
    try:
        from theta_multi_regime_backtrader import MultiRegimeOptionsStrategy, RiskManager
        
        # Test RiskManager
        risk_mgr = RiskManager()
        print("✅ RiskManager initialization successful")
        
        # Test position sizing
        portfolio_value = 25000
        strategy_type = "IRON_CONDOR"
        max_risk = 500
        
        position_size = risk_mgr.calculate_kelly_position(portfolio_value, strategy_type, max_risk)
        print(f"✅ Kelly position calculation: {position_size} contracts")
        
        # Test risk limits
        can_trade = risk_mgr.can_trade(portfolio_value)
        print(f"✅ Risk limit check: {'Can trade' if can_trade else 'Cannot trade'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing strategy components: {e}")
        return False

def test_market_data():
    """Test market data fetching"""
    print("\n🔍 Testing Market Data Fetching...")
    
    try:
        import yfinance as yf
        
        # Test recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        # Fetch SPY data
        spy_df = yf.download('SPY', start=start_date, end=end_date, progress=False)
        print(f"✅ SPY data: {len(spy_df)} days")
        
        # Fetch VIX data
        vix_df = yf.download('^VIX', start=start_date, end=end_date, progress=False)
        print(f"✅ VIX data: {len(vix_df)} days")
        
        if len(spy_df) > 0 and len(vix_df) > 0:
            latest_spy = float(spy_df['Close'].iloc[-1])
            latest_vix = float(vix_df['Close'].iloc[-1])
            print(f"✅ Latest SPY: ${latest_spy:.2f}")
            print(f"✅ Latest VIX: {latest_vix:.2f}")
            return True
        else:
            print("❌ No market data received")
            return False
        
    except Exception as e:
        print(f"❌ Error fetching market data: {e}")
        return False

def test_backtrader_integration():
    """Test basic Backtrader integration"""
    print("\n🔍 Testing Backtrader Integration...")
    
    try:
        import backtrader as bt
        print("✅ Backtrader import successful")
        
        # Test Cerebro initialization
        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(25000)
        print("✅ Cerebro initialization successful")
        
        return True
        
    except ImportError:
        print("❌ Backtrader not installed")
        print("   Install with: pip install backtrader")
        return False
    except Exception as e:
        print(f"❌ Error testing Backtrader: {e}")
        return False

def test_quick_backtest():
    """Run a quick 1-week backtest"""
    print("\n🔍 Running Quick Backtest (1 week)...")
    
    try:
        from theta_multi_regime_backtrader import run_theta_multi_regime_backtest
        
        # Run short backtest
        start_date = '2025-01-02'  # Recent date with likely data
        end_date = '2025-01-10'
        
        print(f"   Testing period: {start_date} to {end_date}")
        
        # This will run the actual backtest
        results = run_theta_multi_regime_backtest(start_date, end_date)
        
        if results:
            print("✅ Quick backtest completed successfully")
            return True
        else:
            print("❌ Quick backtest failed")
            return False
            
    except Exception as e:
        print(f"❌ Error running quick backtest: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🚀 THETADATA BACKTRADER INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("ThetaData Connection", test_theta_connection),
        ("Strategy Components", test_strategy_components),
        ("Market Data", test_market_data),
        ("Backtrader Integration", test_backtrader_integration),
        ("Quick Backtest", test_quick_backtest)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Integration is ready to use.")
    elif passed >= total * 0.8:
        print("⚠️  Most tests passed. Some features may be limited.")
    else:
        print("❌ Multiple test failures. Check your setup.")
    
    print("\n📝 NEXT STEPS:")
    if results.get("ThetaData Connection", False):
        print("   ✅ ThetaData is working - you can use real option data")
    else:
        print("   ⚠️  ThetaData not available - backtest will use simulation")
    
    if all(results[key] for key in ["Strategy Components", "Market Data", "Backtrader Integration"]):
        print("   ✅ Core components working - ready for backtesting")
        print("   🚀 Run: python theta_multi_regime_backtrader.py")
    else:
        print("   ❌ Core components need fixing before running backtests")

if __name__ == "__main__":
    main() 