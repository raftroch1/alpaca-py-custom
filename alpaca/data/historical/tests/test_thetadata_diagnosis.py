#!/usr/bin/env python3
"""
Diagnostic script to discover ThetaData Terminal API endpoints.

This script will test different API paths to find the correct endpoints.
"""

import httpx
import json
from datetime import datetime

def test_endpoints():
    """Test different API endpoints to find the correct ones."""
    print("ThetaData Terminal API Endpoint Discovery")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:25510"
    
    # Common API paths to test
    test_paths = [
        "",                    # Root
        "/",                   # Root with slash
        "/api",               # Common API path
        "/v1",                # Version 1
        "/v2",                # Version 2
        "/rest",              # REST API
        "/option",            # Option endpoints
        "/hist",              # Historical data
        "/snapshot",          # Snapshot data
        "/bulk",              # Bulk data
        "/list",              # List endpoints
        "/quote",             # Quote endpoints
        "/trade",             # Trade endpoints
        "/stream",            # Stream endpoints
    ]
    
    # Test each path
    for path in test_paths:
        print(f"\nTesting: {base_url}{path}")
        try:
            response = httpx.get(f"{base_url}{path}", timeout=5)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"   ✅ SUCCESS!")
                print(f"   Content-Type: {response.headers.get('content-type', 'unknown')}")
                
                # Try to parse as JSON
                try:
                    data = response.json()
                    print(f"   JSON Response: {json.dumps(data, indent=2)[:200]}...")
                except:
                    print(f"   Text Response: {response.text[:200]}...")
            
            elif response.status_code == 404:
                print(f"   ❌ Not Found")
            else:
                print(f"   ⚠️  Status: {response.status_code}")
                print(f"   Response: {response.text[:100]}...")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_common_option_endpoints():
    """Test common option-related endpoints."""
    print("\n" + "=" * 50)
    print("Testing Common Option Endpoints")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:25510"
    
    # Common option endpoints
    endpoints = [
        "/option/list/roots",
        "/option/list/expirations",
        "/option/list/strikes",
        "/option/quote",
        "/option/trade",
        "/option/snapshot",
        "/options/list/roots",
        "/options/list/expirations",
        "/options/list/strikes",
        "/hist/option/quote",
        "/hist/option/trade",
        "/bulk/option/quote",
        "/bulk/option/trade",
        "/snapshot/option",
        "/list/option/roots",
        "/list/option/expirations",
        "/list/option/strikes",
        "/rest/option/list/roots",
        "/rest/option/list/expirations",
        "/api/option/list/roots",
        "/api/option/list/expirations",
        "/v1/option/list/roots",
        "/v1/option/list/expirations",
    ]
    
    for endpoint in endpoints:
        print(f"\nTesting: {base_url}{endpoint}")
        try:
            response = httpx.get(f"{base_url}{endpoint}", timeout=5)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"   ✅ SUCCESS!")
                try:
                    data = response.json()
                    print(f"   JSON: {json.dumps(data, indent=2)[:300]}...")
                except:
                    print(f"   Text: {response.text[:200]}...")
            elif response.status_code != 404:
                print(f"   Response: {response.text[:100]}...")
                
        except Exception as e:
            print(f"   Error: {e}")

def test_with_parameters():
    """Test endpoints with common parameters."""
    print("\n" + "=" * 50)
    print("Testing Endpoints with Parameters")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:25510"
    
    # Test with parameters
    test_cases = [
        ("/", {"symbol": "SPY"}),
        ("/", {"root": "SPY"}),
        ("/option", {"symbol": "SPY"}),
        ("/option", {"root": "SPY"}),
        ("/api", {"symbol": "SPY"}),
        ("/api", {"root": "SPY"}),
        ("/list", {"symbol": "SPY"}),
        ("/list", {"root": "SPY"}),
        ("/quote", {"symbol": "SPY"}),
        ("/quote", {"root": "SPY"}),
        ("/snapshot", {"symbol": "SPY"}),
        ("/snapshot", {"root": "SPY"}),
    ]
    
    for endpoint, params in test_cases:
        print(f"\nTesting: {base_url}{endpoint} with params: {params}")
        try:
            response = httpx.get(f"{base_url}{endpoint}", params=params, timeout=5)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"   ✅ SUCCESS!")
                try:
                    data = response.json()
                    print(f"   JSON: {json.dumps(data, indent=2)[:300]}...")
                except:
                    print(f"   Text: {response.text[:200]}...")
            elif response.status_code != 404:
                print(f"   Response: {response.text[:100]}...")
                
        except Exception as e:
            print(f"   Error: {e}")

def check_terminal_info():
    """Check if we can get terminal information."""
    print("\n" + "=" * 50)
    print("Checking Terminal Information")
    print("=" * 50)
    
    base_url = "http://127.0.0.1:25510"
    
    # Info endpoints
    info_endpoints = [
        "/info",
        "/status",
        "/health",
        "/version",
        "/help",
        "/docs",
        "/api-docs",
        "/swagger",
        "/openapi",
        "/",
    ]
    
    for endpoint in info_endpoints:
        print(f"\nTesting: {base_url}{endpoint}")
        try:
            response = httpx.get(f"{base_url}{endpoint}", timeout=5)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"   ✅ SUCCESS!")
                content_type = response.headers.get('content-type', '')
                if 'json' in content_type.lower():
                    try:
                        data = response.json()
                        print(f"   JSON: {json.dumps(data, indent=2)[:500]}...")
                    except:
                        print(f"   Text: {response.text[:300]}...")
                else:
                    print(f"   Content: {response.text[:300]}...")
            elif response.status_code != 404:
                print(f"   Response: {response.text[:100]}...")
                
        except Exception as e:
            print(f"   Error: {e}")

def main():
    """Main diagnostic function."""
    print("Starting ThetaData Terminal API Diagnostics...")
    print(f"Time: {datetime.now()}")
    print()
    
    # Test 1: Basic endpoints
    test_endpoints()
    
    # Test 2: Option-specific endpoints
    test_common_option_endpoints()
    
    # Test 3: Endpoints with parameters
    test_with_parameters()
    
    # Test 4: Terminal info
    check_terminal_info()
    
    print("\n" + "=" * 50)
    print("Diagnostic Complete!")
    print("Look for endpoints that returned Status: 200 (SUCCESS)")
    print("=" * 50)

if __name__ == "__main__":
    main() 