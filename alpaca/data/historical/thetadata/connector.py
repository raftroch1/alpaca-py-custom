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
        """Test ThetaData connection with a known working request."""
        try:
            url = f"{self.base_url}/v2/hist/option/eod"
            params = {
                'root': 'SPY',
                'exp': '20240705',
                'strike': '535000',
                'right': 'P',
                'start_date': '20240705',
                'end_date': '20240705'
            }
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return 'response' in data and len(data['response']) > 0
            return False
        except Exception:
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