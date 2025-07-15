#!/usr/bin/env python3
"""
Debug script to examine ThetaData response structure
"""

import sys
import os
import json
import pprint
from datetime import datetime

# Add the thetadata directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "thetadata"))
from client import WorkingThetaDataClient

def debug_thetadata_response():
    """Debug ThetaData response structure"""
    print("🔍 DEBUGGING THETADATA RESPONSE STRUCTURE")
    print("=" * 60)
    
    # Initialize client
    client = WorkingThetaDataClient()
    
    # Test 1: Check connection
    print("\n1. Testing connection...")
    roots = client.list_option_roots()
    if roots:
        print(f"✅ Connection successful")
        print(f"   Response type: {type(roots)}")
        print(f"   Response preview: {str(roots)[:200]}...")
    else:
        print("❌ Connection failed")
        return
    
    # Test 2: List option contracts for a specific date
    print("\n2. Testing option contracts listing...")
    contracts = client.list_option_contracts("20240613", "SPY")
    if contracts:
        print(f"✅ Contracts found")
        print(f"   Response type: {type(contracts)}")
        print(f"   Response preview: {str(contracts)[:300]}...")
    else:
        print("❌ No contracts found")
    
    # Test 3: Try to get historical option data
    print("\n3. Testing historical option data...")
    
    # Try a recent date that should have data
    recent_date = "20240613"
    strike = 530000  # $530 strike in cents
    
    print(f"   Fetching SPY {recent_date} ${strike/1000}P...")
    
    response = client.get_historical_option_trade_quote(
        root="SPY",
        exp=recent_date,
        strike=strike,
        right="P",
        start_date=recent_date,
        end_date=recent_date
    )
    
    if response:
        print(f"✅ Response received")
        print(f"   Response type: {type(response)}")
        
        if isinstance(response, dict):
            print(f"   Response keys: {list(response.keys())}")
            
            # Pretty print the response structure
            print("\n   DETAILED RESPONSE STRUCTURE:")
            print("   " + "="*50)
            pprint.pprint(response, indent=4, width=100)
            
            # Check for common fields
            if 'response' in response:
                print(f"\n   Response['response'] type: {type(response['response'])}")
                if isinstance(response['response'], list) and len(response['response']) > 0:
                    first_item = response['response'][0]
                    print(f"   First item type: {type(first_item)}")
                    if isinstance(first_item, dict):
                        print(f"   First item keys: {list(first_item.keys())}")
                        pprint.pprint(first_item, indent=6, width=100)
                    else:
                        print(f"   First item value: {first_item}")
                        
        elif isinstance(response, str):
            print(f"   Response (CSV/Text):")
            print(f"   {response[:500]}...")
            
            # Try to parse as CSV
            lines = response.strip().split('\n')
            print(f"   CSV lines: {len(lines)}")
            if len(lines) > 0:
                print(f"   Header: {lines[0]}")
                if len(lines) > 1:
                    print(f"   First data row: {lines[1]}")
                    
        else:
            print(f"   Response: {response}")
    else:
        print("❌ No response received")
    
    # Test 4: Try different date
    print("\n4. Testing different date...")
    test_date = "20241201"  # More recent date
    
    response2 = client.get_historical_option_trade_quote(
        root="SPY",
        exp=test_date,
        strike=550000,  # $550 strike
        right="P",
        start_date=test_date,
        end_date=test_date
    )
    
    if response2:
        print(f"✅ Response received for {test_date}")
        print(f"   Response type: {type(response2)}")
        if isinstance(response2, dict):
            print(f"   Response keys: {list(response2.keys())}")
        elif isinstance(response2, str):
            print(f"   Response preview: {response2[:200]}...")
    else:
        print(f"❌ No response for {test_date}")
    
    print("\n" + "="*60)
    print("🎯 DEBUGGING COMPLETE")

if __name__ == "__main__":
    debug_thetadata_response() 