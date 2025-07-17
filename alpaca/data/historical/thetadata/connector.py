#!/usr/bin/env python3
"""
ThetaData Connector
Reusable, robust connector for ThetaData API (historical options data)

- Proven working format (strike in thousandths, YYYYMMDD dates)
- Caching for efficiency
- Safe float conversion utility
- To be imported by all strategies/backtests needing ThetaData
"""

import requests
from datetime import datetime
from typing import Optional, Dict, Any

class ThetaDataConnector:
    """
    Shared connector for ThetaData API (real option price fetching)
    Usage:
        from alpaca.data.historical.thetadata.connector import ThetaDataConnector
        td = ThetaDataConnector()
        price = td.get_option_price('SPY', '2025-01-02', 535.0, 'P')
    """
    def __init__(self, base_url: str = "http://127.0.0.1:25510"):
        self.base_url = base_url
        self.session = requests.Session()
        self.cache: Dict[str, Optional[float]] = {}

    def test_connection(self) -> bool:
        """Test ThetaData connection with comprehensive status checking."""
        try:
            url = f"{self.base_url}/v2/hist/option/eod"
            # Use a more recent date that should have data
            params = {
                'root': 'SPY',
                'exp': '20250117',  # Recent expiration date
                'strike': '600000',  # $600 strike in thousandths
                'right': 'P',
                'start_date': '20250115',  # Recent trading date
                'end_date': '20250115'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'response' in data and len(data['response']) > 0:
                        print("✅ ThetaData connection successful")
                        print("✅ Options data accessible")
                        return True
                    else:
                        print("⚠️  ThetaData connected but no data returned (may be normal for recent dates)")
                        return False
                except Exception as e:
                    print(f"❌ JSON parsing error: {e}")
                    print(f"📄 Raw response: {response.text[:200]}...")
                    return False
            elif response.status_code == 474:
                print("❌ ThetaData Terminal connection error:")
                print("   Status 474: Connection lost to Theta Data MDDS")
                print("   🔧 SOLUTION:")
                print("   1. Restart ThetaData Terminal:")
                print("      - Close the current ThetaTerminal.jar process")
                print("      - Navigate to alpaca/data/historical/thetadata/")
                print("      - Run: java -jar ThetaTerminal.jar [your_email] [your_password]")
                print("   2. Wait for connection to establish (may take 30-60 seconds)")
                print("   3. Verify your ThetaData subscription includes options data")
                print("   4. Check your internet connection")
                return False
            else:
                print(f"❌ HTTP {response.status_code}: {response.text[:200]}...")
                return False
        except Exception as e:
            print(f"❌ Connection test error: {e}")
            print("   Make sure ThetaData Terminal is running on port 25510")
            return False

    def get_option_price(self, symbol: str, date: str, strike: float, right: str) -> Optional[float]:
        """
        Fetch option close price from ThetaData API.
        Args:
            symbol: Root symbol (e.g., 'SPY')
            date: Date in YYYY-MM-DD format
            strike: Strike price as float
            right: 'C' for call, 'P' for put
        Returns:
            Option close price (float) or None if unavailable
        """
        cache_key = f"{symbol}_{date}_{strike}_{right}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        try:
            exp_date = datetime.strptime(date, '%Y-%m-%d').strftime('%Y%m%d')
            strike_thousands = int(float(strike) * 1000)
            url = f"{self.base_url}/v2/hist/option/eod"
            params = {
                'root': symbol,
                'exp': exp_date,
                'strike': str(strike_thousands),
                'right': right,
                'start_date': exp_date,
                'end_date': exp_date
            }
            response = self.session.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'response' in data and len(data['response']) > 0:
                    option_data = data['response'][0]
                    close_price = self.safe_float(option_data[5])
                    self.cache[cache_key] = close_price
                    return close_price
            self.cache[cache_key] = None
            return None
        except Exception:
            self.cache[cache_key] = None
            return None

    @staticmethod
    def safe_float(val: Any) -> Optional[float]:
        """Safely convert a value to float, return None if not possible."""
        try:
            return float(val)
        except (TypeError, ValueError):
            return None 