"""
ThetaData API Client for Historical Options Data

This client provides methods to fetch historical options data from ThetaData Terminal,
including expired contracts which are crucial for backtesting 0DTE strategies.
"""

import httpx
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import time
import logging

logger = logging.getLogger(__name__)


class ThetaDataClient:
    """
    Client for fetching historical options data from ThetaData Terminal.
    
    ThetaData provides comprehensive historical options data including:
    - Historical option chains
    - Option quotes and trades
    - Expired contracts data
    - Intraday data
    
    Requires ThetaData Terminal to be running on localhost:25510
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:25510/v2", timeout: int = 60):
        """
        Initialize ThetaData client.
        
        Args:
            base_url: Base URL for ThetaData Terminal API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
        
    def check_connection(self) -> bool:
        """Check if ThetaData Terminal is running and accessible."""
        try:
            response = self.client.get(f"{self.base_url}/option/list/roots")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"ThetaData connection failed: {e}")
            return False
    
    def get_option_expirations(self, root: str = "SPY") -> List[str]:
        """
        Get all available expiration dates for an underlying symbol.
        
        Args:
            root: Underlying symbol (e.g., "SPY")
            
        Returns:
            List of expiration dates in YYYY-MM-DD format
        """
        try:
            url = f"{self.base_url}/option/list/expirations"
            response = self.client.get(url, params={"root": root})
            response.raise_for_status()
            
            data = response.json()
            if "response" in data and "expirations" in data["response"]:
                return data["response"]["expirations"]
            return []
        except Exception as e:
            logger.error(f"Failed to get option expirations: {e}")
            return []
    
    def get_option_strikes(self, root: str = "SPY", exp: Optional[str] = None) -> List[float]:
        """
        Get all available strike prices for an underlying and expiration.
        
        Args:
            root: Underlying symbol
            exp: Expiration date in YYYY-MM-DD format
            
        Returns:
            List of strike prices
        """
        try:
            url = f"{self.base_url}/option/list/strikes"
            params = {"root": root}
            if exp:
                params["exp"] = exp
                
            response = self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if "response" in data and "strikes" in data["response"]:
                return data["response"]["strikes"]
            return []
        except Exception as e:
            logger.error(f"Failed to get option strikes: {e}")
            return []
    
    def get_option_chain_hist(self, root: str, date: str, exp: Optional[str] = None) -> pd.DataFrame:
        """
        Get historical option chain data for a specific date.
        
        Args:
            root: Underlying symbol (e.g., "SPY")
            date: Date in YYYY-MM-DD format
            exp: Optional expiration filter in YYYY-MM-DD format
            
        Returns:
            DataFrame with option chain data including bid/ask/last prices
        """
        try:
            url = f"{self.base_url}/option/quote/hist"
            params = {
                "root": root,
                "date": date,
                "ivl": "0",  # End of day data
                "rth": "true"  # Regular trading hours
            }
            
            if exp:
                params["exp"] = exp
                
            response = self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if "response" in data:
                # Convert to DataFrame
                df = pd.DataFrame(data["response"])
                if not df.empty:
                    # Add parsed columns
                    df["query_date"] = pd.to_datetime(date)
                    df["expiration"] = pd.to_datetime(df["exp"])
                    df["type"] = df["right"].str.lower()
                    df["strike"] = df["strike"] / 1000.0  # ThetaData uses millistrikes
                    df["symbol"] = df["contract"]
                    
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to get option chain hist: {e}")
            return pd.DataFrame()
    
    def get_0dte_chain_hist(self, root: str, date: str) -> pd.DataFrame:
        """
        Get 0DTE option chain data for a specific date.
        
        Args:
            root: Underlying symbol
            date: Date in YYYY-MM-DD format
            
        Returns:
            DataFrame with 0DTE option chain data
        """
        # For 0DTE, the expiration date is the same as the query date
        return self.get_option_chain_hist(root, date, exp=date)
    
    def get_option_trades_hist(self, root: str, date: str, exp: Optional[str] = None) -> pd.DataFrame:
        """
        Get historical option trades data for a specific date.
        
        Args:
            root: Underlying symbol
            date: Date in YYYY-MM-DD format
            exp: Optional expiration filter
            
        Returns:
            DataFrame with option trades data
        """
        try:
            url = f"{self.base_url}/option/trade/hist"
            params = {
                "root": root,
                "date": date,
                "ivl": "0",
                "rth": "true"
            }
            
            if exp:
                params["exp"] = exp
                
            response = self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            if "response" in data:
                df = pd.DataFrame(data["response"])
                if not df.empty:
                    df["query_date"] = pd.to_datetime(date)
                    df["expiration"] = pd.to_datetime(df["exp"])
                    df["type"] = df["right"].str.lower()
                    df["strike"] = df["strike"] / 1000.0
                    df["symbol"] = df["contract"]
                    
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to get option trades hist: {e}")
            return pd.DataFrame()
    
    def fetch_historical_0dte_data(self, root: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch comprehensive 0DTE historical data for a date range.
        
        Args:
            root: Underlying symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with all 0DTE data for the date range
        """
        all_data = []
        
        # Generate business days
        date_range = pd.date_range(start_date, end_date, freq='B')
        
        for date in date_range:
            date_str = date.strftime('%Y-%m-%d')
            logger.info(f"Fetching 0DTE data for {date_str}")
            
            # Get 0DTE chain data
            chain_df = self.get_0dte_chain_hist(root, date_str)
            
            if not chain_df.empty:
                all_data.append(chain_df)
                logger.info(f"  Found {len(chain_df)} 0DTE contracts")
            else:
                logger.warning(f"  No 0DTE contracts found for {date_str}")
                
            # Rate limiting
            time.sleep(0.1)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()


def test_thetadata_connection():
    """Test ThetaData connection and basic functionality."""
    with ThetaDataClient() as client:
        print("Testing ThetaData connection...")
        
        if not client.check_connection():
            print("❌ ThetaData Terminal not accessible. Please ensure it's running.")
            return False
            
        print("✅ ThetaData Terminal connected successfully")
        
        # Test basic functionality
        print("\nTesting basic functionality...")
        expirations = client.get_option_expirations("SPY")
        print(f"Found {len(expirations)} expiration dates for SPY")
        
        if expirations:
            print(f"Recent expirations: {expirations[:5]}")
            
        return True


if __name__ == "__main__":
    test_thetadata_connection() 