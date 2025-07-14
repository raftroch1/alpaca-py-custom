"""
ThetaData REST API Client for Historical Options Data

This client uses ThetaData Terminal's REST API to fetch historical options data.
The REST API is the recommended approach as the Python socket API is deprecated.

Based on ThetaData Terminal REST API:
- Base URL: http://127.0.0.1:25510
- Endpoints: /quote, /trade, /ohlc, /greeks, /snapshot, etc.
- Documentation: https://www.thetadata.net/docs
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import time
import logging
from urllib.parse import urljoin

try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)

class ThetaDataRestClient:
    """
    REST API client for ThetaData Terminal.
    
    This client uses ThetaData's REST API to fetch historical options data
    including expired contracts, quotes, trades, and Greeks.
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:25510", timeout: int = 30):
        """
        Initialize ThetaData REST client.
        
        Args:
            base_url: ThetaData Terminal base URL (default: http://127.0.0.1:25510)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
        """
        Make a request to ThetaData REST API.
        
        Args:
            endpoint: API endpoint (e.g., '/quote', '/trade')
            params: Query parameters
            
        Returns:
            JSON response as dict or None if error
        """
        try:
            url = urljoin(self.base_url, endpoint.lstrip('/'))
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            # ThetaData returns different content types
            if response.headers.get('content-type', '').startswith('application/json'):
                return response.json()
            else:
                # Some endpoints return CSV or other formats
                return {'data': response.text, 'content_type': response.headers.get('content-type')}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {endpoint}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {endpoint}: {e}")
            return None
    
    def get_quote(self, root: str, exp: str, strike: float, right: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Get historical quotes for an options contract.
        
        Args:
            root: Underlying symbol (e.g., 'SPY')
            exp: Expiration date in YYYY-MM-DD format
            strike: Strike price
            right: Option right ('C' for call, 'P' for put)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with quote data or None if error
        """
        params = {
            'root': root,
            'exp': exp,
            'strike': strike,
            'right': right,
            'start_date': start_date,
            'end_date': end_date
        }
        
        response = self._make_request('/quote', params)
        if response:
            return self._parse_response_to_dataframe(response)
        return None
    
    def get_trade(self, root: str, exp: str, strike: float, right: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Get historical trades for an options contract.
        
        Args:
            root: Underlying symbol (e.g., 'SPY')
            exp: Expiration date in YYYY-MM-DD format
            strike: Strike price
            right: Option right ('C' for call, 'P' for put)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with trade data or None if error
        """
        params = {
            'root': root,
            'exp': exp,
            'strike': strike,
            'right': right,
            'start_date': start_date,
            'end_date': end_date
        }
        
        response = self._make_request('/trade', params)
        if response:
            return self._parse_response_to_dataframe(response)
        return None
    
    def get_ohlc(self, root: str, exp: str, strike: float, right: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Get historical OHLC data for an options contract.
        
        Args:
            root: Underlying symbol (e.g., 'SPY')
            exp: Expiration date in YYYY-MM-DD format
            strike: Strike price
            right: Option right ('C' for call, 'P' for put)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with OHLC data or None if error
        """
        params = {
            'root': root,
            'exp': exp,
            'strike': strike,
            'right': right,
            'start_date': start_date,
            'end_date': end_date
        }
        
        response = self._make_request('/ohlc', params)
        if response:
            return self._parse_response_to_dataframe(response)
        return None
    
    def get_greeks(self, root: str, exp: str, strike: float, right: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Get historical Greeks for an options contract.
        
        Args:
            root: Underlying symbol (e.g., 'SPY')
            exp: Expiration date in YYYY-MM-DD format
            strike: Strike price
            right: Option right ('C' for call, 'P' for put)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with Greeks data or None if error
        """
        params = {
            'root': root,
            'exp': exp,
            'strike': strike,
            'right': right,
            'start_date': start_date,
            'end_date': end_date
        }
        
        response = self._make_request('/greeks', params)
        if response:
            return self._parse_response_to_dataframe(response)
        return None
    
    def get_snapshot(self, root: str, exp: str = None) -> Optional[pd.DataFrame]:
        """
        Get snapshot data for options contracts.
        
        Args:
            root: Underlying symbol (e.g., 'SPY')
            exp: Expiration date in YYYY-MM-DD format (optional)
            
        Returns:
            DataFrame with snapshot data or None if error
        """
        params = {'root': root}
        if exp:
            params['exp'] = exp
            
        response = self._make_request('/snapshot', params)
        if response:
            return self._parse_response_to_dataframe(response)
        return None
    
    def list_contracts(self, root: str, exp: str = None) -> Optional[List[Dict]]:
        """
        List available contracts for a root symbol.
        
        Args:
            root: Underlying symbol (e.g., 'SPY')
            exp: Expiration date in YYYY-MM-DD format (optional)
            
        Returns:
            List of contract dictionaries or None if error
        """
        params = {'root': root}
        if exp:
            params['exp'] = exp
            
        response = self._make_request('/list/contracts', params)
        if response and isinstance(response, dict):
            return response.get('contracts', [])
        return None
    
    def list_expirations(self, root: str) -> Optional[List[str]]:
        """
        List available expiration dates for a root symbol.
        
        Args:
            root: Underlying symbol (e.g., 'SPY')
            
        Returns:
            List of expiration dates or None if error
        """
        params = {'root': root}
        response = self._make_request('/list/expirations', params)
        if response and isinstance(response, dict):
            return response.get('expirations', [])
        return None
    
    def list_strikes(self, root: str, exp: str) -> Optional[List[float]]:
        """
        List available strike prices for a root symbol and expiration.
        
        Args:
            root: Underlying symbol (e.g., 'SPY')
            exp: Expiration date in YYYY-MM-DD format
            
        Returns:
            List of strike prices or None if error
        """
        params = {'root': root, 'exp': exp}
        response = self._make_request('/list/strikes', params)
        if response and isinstance(response, dict):
            return response.get('strikes', [])
        return None
    
    def _parse_response_to_dataframe(self, response: Dict) -> Optional[pd.DataFrame]:
        """
        Parse ThetaData response to pandas DataFrame.
        
        Args:
            response: Response dictionary from ThetaData API
            
        Returns:
            DataFrame or None if parsing fails
        """
        try:
            if 'data' in response:
                # Handle CSV response
                content_type = response.get('content_type', '')
                if 'csv' in content_type:
                    from io import StringIO
                    return pd.read_csv(StringIO(response['data']))
                elif 'json' in content_type:
                    return pd.DataFrame(response['data'])
                else:
                    # Try to parse as CSV by default
                    from io import StringIO
                    return pd.read_csv(StringIO(response['data']))
            else:
                # Handle JSON response
                return pd.DataFrame(response)
                
        except Exception as e:
            logger.error(f"Error parsing response to DataFrame: {e}")
            return None
    
    def test_connection(self) -> bool:
        """
        Test connection to ThetaData Terminal.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self._make_request('/list/roots')
            return response is not None
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_0dte_contracts(self, root: str, date: str) -> Optional[pd.DataFrame]:
        """
        Get 0DTE (zero days to expiration) contracts for a specific date.
        
        Args:
            root: Underlying symbol (e.g., 'SPY')
            date: Date in YYYY-MM-DD format
            
        Returns:
            DataFrame with 0DTE contracts or None if error
        """
        # For 0DTE, expiration date is the same as the trade date
        exp_date = date
        
        # Get all contracts for this expiration
        contracts = self.list_contracts(root, exp_date)
        if not contracts:
            return None
            
        # Get quotes for all contracts
        all_data = []
        for contract in contracts:
            strike = contract.get('strike')
            right = contract.get('right')
            
            if strike and right:
                quote_data = self.get_quote(root, exp_date, strike, right, date, date)
                if quote_data is not None and not quote_data.empty:
                    quote_data['strike'] = strike
                    quote_data['right'] = right
                    quote_data['expiration'] = exp_date
                    all_data.append(quote_data)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return None


def test_thetadata_rest_connection():
    """Test the ThetaData REST API connection."""
    print("Testing ThetaData REST API Connection")
    print("=" * 50)
    
    client = ThetaDataRestClient()
    
    # Test basic connection
    print("🔍 Testing basic connection...")
    if client.test_connection():
        print("✅ Connected to ThetaData Terminal REST API")
    else:
        print("❌ Failed to connect to ThetaData Terminal")
        return False
    
    # Test list expirations
    print("\n🔍 Testing list expirations for SPY...")
    expirations = client.list_expirations("SPY")
    if expirations:
        print(f"✅ Found {len(expirations)} expirations for SPY")
        print(f"First few expirations: {expirations[:5]}")
    else:
        print("❌ Failed to get expirations")
    
    # Test list strikes
    if expirations:
        print(f"\n🔍 Testing list strikes for SPY {expirations[0]}...")
        strikes = client.list_strikes("SPY", expirations[0])
        if strikes:
            print(f"✅ Found {len(strikes)} strikes for SPY {expirations[0]}")
            print(f"First few strikes: {strikes[:5]}")
        else:
            print("❌ Failed to get strikes")
    
    # Test quote data
    print("\n🔍 Testing quote data...")
    quote_data = client.get_quote("SPY", "2024-06-14", 420.0, "C", "2024-06-13", "2024-06-14")
    if quote_data is not None:
        print(f"✅ Got quote data: {quote_data.shape[0]} rows")
        print("Sample data:")
        print(quote_data.head())
    else:
        print("❌ Failed to get quote data")
    
    return True


if __name__ == "__main__":
    test_thetadata_rest_connection() 