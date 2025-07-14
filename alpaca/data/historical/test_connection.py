#!/usr/bin/env python3
"""
Test script to verify ThetaData Terminal connection and data fetching.

This script will:
1. Test ThetaData Terminal connection
2. Test basic data fetching functionality
3. Verify data format and structure
4. Test 0DTE data fetching for a specific date
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from thetadata_client import ThetaDataClient
    import pandas as pd
    import httpx
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install: pip install httpx pandas")
    sys.exit(1)

def test_basic_connection():
    """Test basic ThetaData Terminal connection."""
    print("=" * 50)
    print("Testing ThetaData Terminal Connection")
    print("=" * 50)
    
    # Test direct connection
    try:
        response = httpx.get("http://127.0.0.1:25510/v2/option/list/roots", timeout=10)
        print(f"✅ Direct connection successful (Status: {response.status_code})")
        print(f"Response preview: {response.text[:200]}...")
    except Exception as e:
        print(f"❌ Direct connection failed: {e}")
        return False
    
    # Test ThetaData client
    try:
        with ThetaDataClient() as client:
            if client.check_connection():
                print("✅ ThetaData client connection successful")
                return True
            else:
                print("❌ ThetaData client connection failed")
                return False
    except Exception as e:
        print(f"❌ ThetaData client error: {e}")
        return False

def test_data_fetching():
    """Test data fetching functionality."""
    print("\n" + "=" * 50)
    print("Testing Data Fetching")
    print("=" * 50)
    
    with ThetaDataClient() as client:
        # Test 1: Get expirations
        print("\n1. Testing option expirations...")
        expirations = client.get_option_expirations("SPY")
        print(f"   Found {len(expirations)} expirations")
        if expirations:
            print(f"   Recent expirations: {expirations[:5]}")
        
        # Test 2: Get strikes
        print("\n2. Testing option strikes...")
        strikes = client.get_option_strikes("SPY")
        print(f"   Found {len(strikes)} strikes")
        if strikes:
            print(f"   Sample strikes: {strikes[:10]}")
        
        # Test 3: Get historical data for a specific date
        print("\n3. Testing historical 0DTE data...")
        test_date = "2024-06-14"  # Friday
        print(f"   Fetching 0DTE data for {test_date}...")
        
        df = client.get_0dte_chain_hist("SPY", test_date)
        if not df.empty:
            print(f"   ✅ Found {len(df)} 0DTE contracts")
            print(f"   Columns: {list(df.columns)}")
            print(f"   Sample data:")
            print(df.head(3))
        else:
            print(f"   ❌ No 0DTE data found for {test_date}")
        
        return not df.empty

def test_date_range_fetching():
    """Test fetching data for a date range."""
    print("\n" + "=" * 50)
    print("Testing Date Range Fetching")
    print("=" * 50)
    
    with ThetaDataClient() as client:
        # Test small date range
        start_date = "2024-06-14"
        end_date = "2024-06-17"  # Friday to Monday
        
        print(f"Fetching 0DTE data from {start_date} to {end_date}...")
        df = client.fetch_historical_0dte_data("SPY", start_date, end_date)
        
        if not df.empty:
            print(f"✅ Successfully fetched {len(df)} records")
            print(f"Date range: {df['query_date'].min()} to {df['query_date'].max()}")
            print(f"Unique dates: {df['query_date'].dt.date.nunique()}")
            print(f"Sample data:")
            print(df.head())
            
            # Save sample data
            sample_file = "theta_sample_data.csv"
            df.to_csv(sample_file, index=False)
            print(f"Saved sample data to {sample_file}")
            
        else:
            print("❌ No data fetched for date range")
        
        return not df.empty

def test_specific_endpoints():
    """Test specific ThetaData endpoints."""
    print("\n" + "=" * 50)
    print("Testing Specific Endpoints")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:25510/v2"
    
    # Test endpoints
    endpoints = [
        ("/option/list/roots", {"limit": 10}),
        ("/option/list/expirations", {"root": "SPY"}),
        ("/option/list/strikes", {"root": "SPY"}),
        ("/option/quote/hist", {"root": "SPY", "date": "2024-06-14", "ivl": "0"}),
    ]
    
    for endpoint, params in endpoints:
        print(f"\nTesting {endpoint}...")
        try:
            response = httpx.get(f"{base_url}{endpoint}", params=params, timeout=30)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if "response" in data:
                    response_data = data["response"]
                    if isinstance(response_data, list):
                        print(f"   Response: {len(response_data)} items")
                        if response_data:
                            print(f"   Sample: {response_data[0]}")
                    elif isinstance(response_data, dict):
                        print(f"   Response keys: {list(response_data.keys())}")
                    else:
                        print(f"   Response: {response_data}")
                else:
                    print(f"   Raw response: {response.text[:200]}...")
            else:
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"   Error: {e}")

def main():
    """Main test function."""
    print("ThetaData Terminal Connection Test")
    print("Make sure ThetaData Terminal is running on localhost:25510")
    print()
    
    # Test 1: Basic connection
    if not test_basic_connection():
        print("\n❌ Basic connection failed. Please check that ThetaData Terminal is running.")
        return
    
    # Test 2: Data fetching
    if not test_data_fetching():
        print("\n❌ Data fetching failed. Check ThetaData Terminal and data availability.")
        return
    
    # Test 3: Date range fetching
    if not test_date_range_fetching():
        print("\n❌ Date range fetching failed.")
        return
    
    # Test 4: Specific endpoints
    test_specific_endpoints()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed successfully!")
    print("ThetaData Terminal is working correctly.")
    print("You can now run the improved backtest script.")
    print("=" * 50)

if __name__ == "__main__":
    main() 