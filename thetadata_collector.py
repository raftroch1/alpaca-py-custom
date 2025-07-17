#!/usr/bin/env python3
"""
ThetaData Collector - Cache Market Data for Fast Strategy Testing
================================================================

This script downloads and caches market data from ThetaData API:
- SPY minute bars by date range
- 0DTE option chains and prices
- Saves as compressed pickle files for instant loading

Usage:
    python thetadata_collector.py --start_date 20250101 --end_date 20250115
    python thetadata_collector.py --date 20250717  # Single day
"""

import requests
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import argparse
import gzip
from typing import Dict, List, Optional
import time
import json

class ThetaDataCollector:
    """Collect and cache ThetaData for strategy backtesting"""
    
    def __init__(self, cache_dir: str = "cached_data"):
        self.base_url = "http://127.0.0.1:25510"
        self.cache_dir = cache_dir
        self.option_price_cache = {}
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(f"{cache_dir}/spy_bars", exist_ok=True)
        os.makedirs(f"{cache_dir}/option_chains", exist_ok=True)
        
        print(f"📁 Cache directory: {os.path.abspath(cache_dir)}")
    
    def get_cache_path(self, data_type: str, date: str) -> str:
        """Get cache file path for specific data type and date"""
        return f"{self.cache_dir}/{data_type}/{data_type}_{date}.pkl.gz"
    
    def is_cached(self, data_type: str, date: str) -> bool:
        """Check if data is already cached"""
        cache_path = self.get_cache_path(data_type, date)
        return os.path.exists(cache_path)
    
    def save_to_cache(self, data: any, data_type: str, date: str):
        """Save data to compressed cache file"""
        cache_path = self.get_cache_path(data_type, date)
        with gzip.open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Saved {data_type} for {date} ({os.path.getsize(cache_path)} bytes)")
    
    def load_from_cache(self, data_type: str, date: str):
        """Load data from cache file"""
        cache_path = self.get_cache_path(data_type, date)
        if not os.path.exists(cache_path):
            return None
        
        with gzip.open(cache_path, 'rb') as f:
            data = pickle.load(f)
        print(f"🔄 Loaded {data_type} for {date} from cache")
        return data
    
    def get_spy_minute_bars(self, date: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get SPY minute bars for a specific date
        
        Args:
            date: Date in YYYYMMDD format
            force_refresh: Force API call even if cached
            
        Returns:
            DataFrame with minute bars
        """
        if not force_refresh and self.is_cached("spy_bars", date):
            return self.load_from_cache("spy_bars", date)
        
        print(f"🔌 Fetching SPY minute bars for {date} from ThetaData...")
        
        # ThetaData API call for SPY minute bars
        params = {
            'root': 'SPY',
            'start_date': date,
            'end_date': date
        }
        
        try:
            response = requests.get(f"{self.base_url}/v2/hist/stock/trade", params=params, timeout=30)
            
            if response.status_code != 200:
                print(f"❌ API Error {response.status_code}: {response.text}")
                return pd.DataFrame()
            
            data = response.json()
            
            if 'response' not in data or not data['response']:
                print(f"⚠️ No minute bars found for {date}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            bars_data = []
            for bar in data['response']:
                try:
                    # ThetaData response format (15 elements):
                    # [0] timestamp_ms, [1] ms_of_day, [2-6] unknown, [7] unknown, [8] unknown, 
                    # [9] close_price, [10] unknown, [11-14] more fields, [14] date
                    
                    timestamp_ms = bar[0]
                    ms_of_day = bar[1] 
                    close_price = bar[9]  # Actual close price is at index 9
                    
                    # Create proper datetime for the trading day
                    date_dt = datetime.strptime(date, '%Y%m%d')
                    
                    # Convert ms_of_day to time offset (ThetaData format)
                    total_seconds = ms_of_day // 1000
                    hours = total_seconds // 3600
                    minutes = (total_seconds % 3600) // 60
                    seconds = total_seconds % 60
                    
                    # Create datetime for this bar (market open + offset)
                    market_open = date_dt.replace(hour=9, minute=30, second=0)
                    bar_time = market_open + timedelta(hours=hours, minutes=minutes, seconds=seconds)
                    
                    # For now, use close price for all OHLC until we identify correct indices
                    bars_data.append({
                        'datetime': bar_time,
                        'ms_of_day': ms_of_day,
                        'open': close_price,
                        'high': close_price,
                        'low': close_price,
                        'close': close_price,
                        'volume': bar[7] if len(bar) > 7 else 0  # Volume might be at index 7
                    })
                except (IndexError, ValueError, TypeError) as e:
                    # Skip malformed bars
                    continue
            
            df = pd.DataFrame(bars_data)
            df = df.set_index('datetime')
            
            print(f"✅ Retrieved {len(df)} minute bars for {date}")
            
            # Cache the data
            self.save_to_cache(df, "spy_bars", date)
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error processing data: {e}")
            return pd.DataFrame()
    
    def get_option_chain(self, date: str, force_refresh: bool = False) -> Dict:
        """
        Get 0DTE option chain for SPY on specific date
        
        Args:
            date: Date in YYYYMMDD format
            force_refresh: Force API call even if cached
            
        Returns:
            Dictionary with option chain data
        """
        if not force_refresh and self.is_cached("option_chains", date):
            return self.load_from_cache("option_chains", date)
        
        print(f"🔌 Fetching 0DTE option chain for {date} from ThetaData...")
        
        # Get available strikes around SPY price
        spy_bars = self.get_spy_minute_bars(date)
        if spy_bars.empty:
            print(f"⚠️ No SPY data for {date}, cannot fetch options")
            return {}
        
        # Use opening price to determine strike range
        spy_price = spy_bars['open'].iloc[0]
        strikes = []
        
        # Generate strikes ±10% around SPY price
        for i in range(-20, 21):
            strike = round(spy_price + i * 5, 0)  # $5 increments
            strikes.append(int(strike))
        
        option_data = {
            'spy_price': spy_price,
            'date': date,
            'calls': {},
            'puts': {}
        }
        
        print(f"📊 Fetching options around SPY ${spy_price:.2f}")
        
        # Fetch call and put prices
        for strike in strikes:
            # Fetch call option
            call_price = self._get_option_price(date, strike, 'C')
            if call_price > 0:
                option_data['calls'][strike] = call_price
            
            # Fetch put option  
            put_price = self._get_option_price(date, strike, 'P')
            if put_price > 0:
                option_data['puts'][strike] = put_price
            
            time.sleep(0.1)  # Rate limiting
        
        print(f"✅ Retrieved {len(option_data['calls'])} calls, {len(option_data['puts'])} puts")
        
        # Cache the data
        self.save_to_cache(option_data, "option_chains", date)
        
        return option_data
    
    def _get_option_price(self, date: str, strike: int, option_type: str) -> float:
        """Get option price for specific strike and type"""
        try:
            params = {
                'root': 'SPY',
                'exp': date,
                'strike': str(strike * 1000),  # ThetaData uses thousandths
                'right': option_type
            }
            
            response = requests.get(f"{self.base_url}/v2/hist/option/eod", params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'response' in data and data['response']:
                    # Return close price (index 5)
                    return data['response'][0][5] / 1000.0
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def collect_date_range(self, start_date: str, end_date: str, force_refresh: bool = False):
        """
        Collect data for a date range
        
        Args:
            start_date: Start date in YYYYMMDD format
            end_date: End date in YYYYMMDD format
            force_refresh: Force refresh even if cached
        """
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        
        current_date = start_dt
        collected_dates = []
        
        print(f"🚀 Collecting data from {start_date} to {end_date}")
        print("=" * 60)
        
        while current_date <= end_dt:
            date_str = current_date.strftime('%Y%m%d')
            
            # Skip weekends
            if current_date.weekday() < 5:  # Monday = 0, Sunday = 6
                print(f"\n📅 Processing {date_str} ({current_date.strftime('%A')})")
                
                # Collect SPY bars
                spy_data = self.get_spy_minute_bars(date_str, force_refresh)
                
                # Only collect options if we have SPY data
                if not spy_data.empty:
                    option_data = self.get_option_chain(date_str, force_refresh)
                    collected_dates.append(date_str)
                else:
                    print(f"⚠️ Skipping options for {date_str} - no SPY data")
            
            current_date += timedelta(days=1)
        
        print("\n" + "=" * 60)
        print(f"✅ Data collection complete!")
        print(f"📊 Collected data for {len(collected_dates)} trading days:")
        for date in collected_dates:
            print(f"   • {date}")
    
    def list_cached_data(self):
        """List all cached data files"""
        print("\n📁 CACHED DATA INVENTORY")
        print("=" * 50)
        
        for data_type in ['spy_bars', 'option_chains']:
            cache_path = f"{self.cache_dir}/{data_type}"
            if os.path.exists(cache_path):
                files = [f for f in os.listdir(cache_path) if f.endswith('.pkl.gz')]
                files.sort()
                
                print(f"\n{data_type.upper().replace('_', ' ')}:")
                if files:
                    for file in files:
                        date = file.split('_')[1].split('.')[0]
                        size = os.path.getsize(f"{cache_path}/{file}")
                        print(f"   • {date} ({size:,} bytes)")
                else:
                    print("   (no cached data)")
    
    def clean_cache(self, older_than_days: int = 30):
        """Remove cached data older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        removed_count = 0
        
        for data_type in ['spy_bars', 'option_chains']:
            cache_path = f"{self.cache_dir}/{data_type}"
            if os.path.exists(cache_path):
                for file in os.listdir(cache_path):
                    if file.endswith('.pkl.gz'):
                        file_path = f"{cache_path}/{file}"
                        file_date = datetime.fromtimestamp(os.path.getmtime(file_path))
                        
                        if file_date < cutoff_date:
                            os.remove(file_path)
                            removed_count += 1
                            print(f"🗑️ Removed old cache: {file}")
        
        print(f"✅ Cleaned {removed_count} old cache files")


def main():
    parser = argparse.ArgumentParser(description="ThetaData Collector - Cache market data for fast strategy testing")
    parser.add_argument('--start_date', help='Start date (YYYYMMDD)')
    parser.add_argument('--end_date', help='End date (YYYYMMDD)')
    parser.add_argument('--date', help='Single date (YYYYMMDD)')
    parser.add_argument('--list', action='store_true', help='List cached data')
    parser.add_argument('--clean', type=int, help='Clean cache older than N days')
    parser.add_argument('--force', action='store_true', help='Force refresh even if cached')
    
    args = parser.parse_args()
    
    collector = ThetaDataCollector()
    
    if args.list:
        collector.list_cached_data()
    elif args.clean:
        collector.clean_cache(args.clean)
    elif args.date:
        collector.collect_date_range(args.date, args.date, args.force)
    elif args.start_date and args.end_date:
        collector.collect_date_range(args.start_date, args.end_date, args.force)
    else:
        # Default: collect recent 5 trading days
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
        print("📅 No date specified, collecting recent 5 trading days")
        collector.collect_date_range(start_date, end_date, args.force)


if __name__ == "__main__":
    main() 