#!/usr/bin/env python3
"""
Working ThetaData REST API Client

This client uses the CORRECT ThetaData Terminal REST API endpoints
based on the actual documentation found in the Obsidian vault.

Key endpoints:
- /v2/hist/option/trade_quote - Historical trades and quotes
- /v2/hist/option/eod - End-of-day data  
- /v2/list/contracts/option/quote - List available contracts
- /v2/list/roots/option - List available symbols
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import time
import logging
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

class WorkingThetaDataClient:
    """
    Working ThetaData REST API client using correct endpoints.
    
    This client uses the actual REST API endpoints documented in ThetaData:
    - Base URL: http://127.0.0.1:25510
    - Historical data: /v2/hist/option/*
    - Contract listing: /v2/list/contracts/option/*
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:25510", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Make HTTP request to ThetaData API."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            logger.info(f"Making request to: {url} with params: {params}")
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            # ThetaData returns different formats (CSV, JSON)
            content_type = response.headers.get('content-type', '').lower()
            logger.info(f"Response content-type: {content_type}")
            logger.info(f"Response status: {response.status_code}")
            
            if 'json' in content_type:
                result = response.json()
                logger.info(f"JSON response type: {type(result)}, keys: {list(result.keys()) if isinstance(result, dict) else 'not dict'}")
                return result
            elif 'csv' in content_type or 'text' in content_type:
                result = response.text
                logger.info(f"Text response preview: {result[:200]}...")
                return result
            else:
                result = response.text
                logger.info(f"Unknown format response preview: {result[:200]}...")
                return result
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test connection to ThetaData Terminal."""
        try:
            # Test with a simple endpoint that should always work
            result = self._make_request("/v2/list/roots/option")
            return result is not None
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def list_option_roots(self) -> Optional[List[str]]:
        """
        List all available option symbols.
        
        Endpoint: /v2/list/roots/option
        Returns: List of symbols like ['SPY', 'AAPL', 'QQQ', ...]
        """
        result = self._make_request("/v2/list/roots/option")
        if result:
            # Handle both JSON and CSV responses
            if isinstance(result, dict):
                # JSON response - extract symbols from the structure
                if 'response' in result:
                    return result['response']
                elif isinstance(result, list):
                    return result
                else:
                    # Look for symbol-like data in the dict
                    return list(result.keys()) if result else None
            elif isinstance(result, str):
                # CSV response - parse as text
                lines = result.strip().split('\n')
                if len(lines) > 1:  # Skip header
                    return [line.split(',')[0] for line in lines[1:] if line.strip()]
                else:
                    # Single line response
                    return [result.strip()] if result.strip() else None
        return None
    
    def list_option_contracts(self, start_date: str, root: Optional[str] = None) -> Optional[Union[List, Dict]]:
        """
        List available option contracts for a specific date.
        
        Args:
            start_date: Date in YYYYMMDD format (e.g., "20240613")
            root: Optional symbol filter (e.g., "SPY")
            
        Endpoint: /v2/list/contracts/option/quote
        Returns: JSON response with available contracts
        """
        params = {"start_date": start_date}
        if root:
            params["root"] = root
            
        result = self._make_request("/v2/list/contracts/option/quote", params)
        
        # Extract the response data from ThetaData's JSON format
        if isinstance(result, dict) and 'response' in result:
            return result['response']
        return result
    
    def get_historical_option_quotes(self, 
                                   root: str,
                                   exp: str,
                                   strike: int,
                                   right: str,
                                   start_date: str,
                                   end_date: str,
                                   ivl: Optional[int] = None) -> Optional[str]:
        """
        Get historical option quotes.
        
        Args:
            root: Symbol (e.g., "SPY")
            exp: Expiration date YYYYMMDD (e.g., "20240613")
            strike: Strike price in cents (e.g., 550000 for $550)
            right: "C" for call, "P" for put
            start_date: Start date YYYYMMDD
            end_date: End date YYYYMMDD
            ivl: Optional interval in milliseconds
            
        Endpoint: /v2/hist/option/quote
        Returns: CSV data with historical quotes
        """
        params = {
            "root": root,
            "exp": exp,
            "strike": strike,
            "right": right,
            "start_date": start_date,
            "end_date": end_date
        }
        
        if ivl:
            params["ivl"] = ivl
            
        return self._make_request("/v2/hist/option/quote", params)
    
    def get_historical_option_trades(self,
                                   root: str,
                                   exp: str,
                                   strike: int,
                                   right: str,
                                   start_date: str,
                                   end_date: str) -> Optional[str]:
        """
        Get historical option trades.
        
        Args:
            root: Symbol (e.g., "SPY")
            exp: Expiration date YYYYMMDD (e.g., "20240613")
            strike: Strike price in cents (e.g., 550000 for $550)
            right: "C" for call, "P" for put
            start_date: Start date YYYYMMDD
            end_date: End date YYYYMMDD
            
        Endpoint: /v2/hist/option/trade
        Returns: CSV data with historical trades
        """
        params = {
            "root": root,
            "exp": exp,
            "strike": strike,
            "right": right,
            "start_date": start_date,
            "end_date": end_date
        }
        
        return self._make_request("/v2/hist/option/trade", params)
    
    def get_historical_option_trade_quote(self,
                                        root: str,
                                        exp: str,
                                        strike: int,
                                        right: str,
                                        start_date: str,
                                        end_date: str) -> Optional[str]:
        """
        Get historical option trades and quotes combined.
        
        This is perfect for 0DTE strategies as it provides both
        trade and quote data in one request.
        
        Args:
            root: Symbol (e.g., "SPY")
            exp: Expiration date YYYYMMDD (e.g., "20240613")
            strike: Strike price in cents (e.g., 550000 for $550)
            right: "C" for call, "P" for put
            start_date: Start date YYYYMMDD
            end_date: End date YYYYMMDD
            
        Endpoint: /v2/hist/option/trade_quote
        Returns: CSV data with historical trades and quotes
        """
        params = {
            "root": root,
            "exp": exp,
            "strike": strike,
            "right": right,
            "start_date": start_date,
            "end_date": end_date
        }
        
        return self._make_request("/v2/hist/option/trade_quote", params)
    
    def get_historical_option_eod(self,
                                root: str,
                                exp: str,
                                strike: int,
                                right: str,
                                start_date: str,
                                end_date: str) -> Optional[str]:
        """
        Get historical end-of-day option data.
        
        Perfect for getting final prices of expired contracts.
        
        Args:
            root: Symbol (e.g., "SPY")
            exp: Expiration date YYYYMMDD (e.g., "20240613")
            strike: Strike price in cents (e.g., 550000 for $550)
            right: "C" for call, "P" for put
            start_date: Start date YYYYMMDD
            end_date: End date YYYYMMDD
            
        Endpoint: /v2/hist/option/eod
        Returns: CSV data with end-of-day prices
        """
        params = {
            "root": root,
            "exp": exp,
            "strike": strike,
            "right": right,
            "start_date": start_date,
            "end_date": end_date
        }
        
        return self._make_request("/v2/hist/option/eod", params)
    
    def get_spy_0dte_data(self, date: str) -> Dict[str, Any]:
        """
        Get 0DTE SPY options data for a specific expiration date.
        
        This method demonstrates how to fetch data for your 0DTE strategy.
        
        Args:
            date: Expiration date in YYYYMMDD format (e.g., "20240613")
            
        Returns:
            Dictionary with contract listings and sample data
        """
        results = {}
        
        # 1. List available SPY contracts for this date
        logger.info(f"Fetching SPY contracts for {date}")
        contracts = self.list_option_contracts(date, "SPY")
        results["contracts"] = contracts
        
        if contracts:
            # Parse contracts to find ATM strikes
            # This is a simplified example - you'd want more sophisticated logic
            if isinstance(contracts, list) and len(contracts) > 0:
                # JSON list response - get first contract as sample
                sample_contract = contracts[0] if contracts else None
                if sample_contract:
                    # Use a reasonable strike for SPY (around $550)
                    sample_strike = 550000  # $550 strike as example
                    
                    # 2. Get historical trade/quote data
                    logger.info(f"Fetching trade/quote data for SPY {date}")
                    trade_quote_data = self.get_historical_option_trade_quote(
                        root="SPY",
                        exp=date,
                        strike=sample_strike,
                        right="C",  # Call option
                        start_date=date,
                        end_date=date
                    )
                    results["sample_trade_quote"] = trade_quote_data
                    
                    # 3. Get EOD data
                    logger.info(f"Fetching EOD data for SPY {date}")
                    eod_data = self.get_historical_option_eod(
                        root="SPY",
                        exp=date,
                        strike=sample_strike,
                        right="C",
                        start_date=date,
                        end_date=date
                    )
                    results["sample_eod"] = eod_data
        
        return results


def main():
    """Test the ThetaData client."""
    print("🔬 Testing ThetaData REST API Client")
    print("=" * 50)
    
    # Initialize client
    client = WorkingThetaDataClient()
    
    # Test connection
    print("Testing connection to ThetaData Terminal...")
    if client.test_connection():
        print("✅ Connected to ThetaData Terminal!")
    else:
        print("❌ Failed to connect to ThetaData Terminal")
        return
    
    # Test listing symbols
    print("\nTesting symbol listing...")
    symbols = client.list_option_roots()
    if symbols:
        print(f"✅ Found {len(symbols)} option symbols")
        print(f"Sample symbols: {symbols[:10]}")  # Show first 10
    else:
        print("❌ Failed to get symbols")
    
    # Test contract listing for a specific date
    test_date = "20240613"  # June 13, 2024 (Thursday - typical 0DTE)
    print(f"\nTesting contract listing for {test_date}...")
    contracts = client.list_option_contracts(test_date, "SPY")
    if contracts:
        if isinstance(contracts, list):
            print(f"✅ Found {len(contracts)} contracts")
            print("Sample contracts:")
            for contract in contracts[:5]:  # Show first 5 contracts
                print(f"  {contract}")
        else:
            print(f"✅ Found contracts data: {type(contracts)}")
    else:
        print("❌ Failed to get contracts")
    
    # Test 0DTE data fetching
    print(f"\nTesting 0DTE data fetching for {test_date}...")
    try:
        spy_data = client.get_spy_0dte_data(test_date)
        print("✅ Successfully fetched 0DTE data structure")
        for key, value in spy_data.items():
            if value:
                print(f"  {key}: {'✅ Has data' if value else '❌ No data'}")
            else:
                print(f"  {key}: ❌ No data")
    except Exception as e:
        print(f"❌ Failed to fetch 0DTE data: {e}")
    
    print("\n🎉 ThetaData client testing complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 