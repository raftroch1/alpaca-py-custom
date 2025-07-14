"""
Improved 0DTE Options Strategy Backtest with ThetaData and Alpaca Integration

This script implements a robust backtesting framework that can fetch historical
options data from both ThetaData and Alpaca APIs, with automatic fallbacks
and proper handling of expired contracts.

Key improvements:
1. ThetaData integration for historical expired options data
2. Alpaca API fallback when ThetaData is unavailable
3. Proper VIX-based regime detection
4. Comprehensive trade logging and analysis
5. Error handling and data validation
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    import yfinance as yf
    from dotenv import load_dotenv
    from thetadata_client import ThetaDataClient
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionChainRequest
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install required packages:")
    print("pip install yfinance python-dotenv httpx pandas numpy")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
class BacktestConfig:
    UNDERLYING = "SPY"
    BACKTEST_START = datetime(2024, 6, 13)
    BACKTEST_END = datetime(2024, 7, 13)
    
    # Output files
    OPTIONS_CSV = "spy_options_historical.csv"
    TRADE_LOG_CSV = "zero_dte_trades_log.csv"
    PERFORMANCE_CSV = "backtest_performance.csv"
    
    # VIX regime thresholds
    VIX_LOW = 17
    VIX_HIGH = 18
    
    # Strategy parameters
    IRON_CONDOR_WING_WIDTH = 5  # strikes
    IRON_BUTTERFLY_WING_WIDTH = 10  # strikes
    POSITION_SIZE = 1  # number of contracts
    
    # Data source preferences
    PREFER_THETADATA = True
    FALLBACK_TO_ALPACA = True


class MultiSourceDataFetcher:
    """
    Fetches historical options data from multiple sources with fallback mechanisms.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.thetadata_client = None
        self.alpaca_client = None
        
        # Initialize clients
        self._init_clients()
    
    def _init_clients(self):
        """Initialize data clients."""
        # Load environment variables
        load_dotenv()
        
        # ThetaData client
        if self.config.PREFER_THETADATA:
            try:
                self.thetadata_client = ThetaDataClient()
                if not self.thetadata_client.check_connection():
                    logger.warning("ThetaData Terminal not accessible")
                    self.thetadata_client = None
                else:
                    logger.info("ThetaData client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize ThetaData client: {e}")
                self.thetadata_client = None
        
        # Alpaca client
        if self.config.FALLBACK_TO_ALPACA:
            try:
                api_key = os.getenv("APCA_API_KEY_ID")
                api_secret = os.getenv("APCA_API_SECRET_KEY")
                
                if api_key and api_secret:
                    self.alpaca_client = OptionHistoricalDataClient(
                        api_key=api_key, 
                        secret_key=api_secret
                    )
                    logger.info("Alpaca client initialized successfully")
                else:
                    logger.warning("Alpaca API credentials not found")
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca client: {e}")
                self.alpaca_client = None
    
    def fetch_historical_options_data(self) -> pd.DataFrame:
        """
        Fetch historical options data using available sources.
        
        Returns:
            DataFrame with historical options data
        """
        # Try ThetaData first
        if self.thetadata_client:
            logger.info("Attempting to fetch data from ThetaData...")
            try:
                return self._fetch_from_thetadata()
            except Exception as e:
                logger.error(f"ThetaData fetch failed: {e}")
        
        # Fallback to Alpaca
        if self.alpaca_client:
            logger.info("Falling back to Alpaca API...")
            try:
                return self._fetch_from_alpaca()
            except Exception as e:
                logger.error(f"Alpaca fetch failed: {e}")
        
        # If both fail, return empty DataFrame
        logger.error("All data sources failed")
        return pd.DataFrame()
    
    def _fetch_from_thetadata(self) -> pd.DataFrame:
        """Fetch data from ThetaData API."""
        start_date = self.config.BACKTEST_START.strftime('%Y-%m-%d')
        end_date = self.config.BACKTEST_END.strftime('%Y-%m-%d')
        
        with self.thetadata_client as client:
            df = client.fetch_historical_0dte_data(
                root=self.config.UNDERLYING,
                start_date=start_date,
                end_date=end_date
            )
            
            if not df.empty:
                logger.info(f"ThetaData: Fetched {len(df)} option records")
                return self._standardize_thetadata_format(df)
            else:
                raise ValueError("No data returned from ThetaData")
    
    def _fetch_from_alpaca(self) -> pd.DataFrame:
        """Fetch data from Alpaca API (limited to recent data)."""
        logger.warning("Alpaca API has limitations for historical expired options data")
        
        # Note: Alpaca's historical options data is limited
        # This is a simplified implementation - real implementation would need
        # to handle Alpaca's specific limitations
        
        all_rows = []
        business_days = pd.date_range(
            self.config.BACKTEST_START, 
            self.config.BACKTEST_END, 
            freq="B"
        )
        
        for query_date in business_days:
            try:
                # Try to get option chain (limited success for historical data)
                chain_req = OptionChainRequest(
                    underlying_symbol=self.config.UNDERLYING,
                    updated_since=query_date
                )
                chain = self.alpaca_client.get_option_chain(chain_req)
                
                # Process chain data (simplified)
                if isinstance(chain, dict) and chain:
                    # This would need more sophisticated processing
                    # for real implementation
                    logger.info(f"Alpaca: Found some contracts for {query_date.date()}")
                    
            except Exception as e:
                logger.warning(f"Alpaca chain fetch failed for {query_date.date()}: {e}")
                continue
        
        # Return empty DataFrame as Alpaca has limitations for historical data
        return pd.DataFrame()
    
    def _standardize_thetadata_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize ThetaData format to match expected schema."""
        if df.empty:
            return df
        
        # Ensure required columns exist
        required_cols = ['query_date', 'expiration', 'type', 'strike', 'bid', 'ask', 'symbol']
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        
        # Convert data types
        df['query_date'] = pd.to_datetime(df['query_date'])
        df['expiration'] = pd.to_datetime(df['expiration'])
        df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
        df['bid'] = pd.to_numeric(df['bid'], errors='coerce')
        df['ask'] = pd.to_numeric(df['ask'], errors='coerce')
        
        return df[required_cols]


class ZeroDTEStrategy:
    """
    0DTE Options Strategy Implementation.
    
    Strategies implemented:
    - High VIX: Iron Condor and Iron Butterfly (premium selling)
    - Low VIX: Diagonal spreads (premium buying)
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trade_log = []
        self.vix_data = None
    
    def load_vix_data(self, options_df: pd.DataFrame):
        """Load VIX data for the backtest period."""
        if options_df.empty:
            self.vix_data = pd.Series(dtype=float)
            return
        
        start_date = options_df["query_date"].min()
        end_date = options_df["query_date"].max()
        
        try:
            vix = yf.download("^VIX", start=start_date, end=end_date)
            self.vix_data = vix["Close"].reindex(
                pd.date_range(start_date, end_date, freq="B")
            ).ffill()
            logger.info(f"Loaded VIX data from {start_date} to {end_date}")
        except Exception as e:
            logger.error(f"Failed to load VIX data: {e}")
            self.vix_data = pd.Series(dtype=float)
    
    def get_market_regime(self, date: datetime) -> str:
        """Determine market regime based on VIX level."""
        try:
            vix_level = float(self.vix_data.loc[date])
            if vix_level > self.config.VIX_HIGH:
                return "HIGH_VOL"
            elif vix_level < self.config.VIX_LOW:
                return "LOW_VOL"
            else:
                return "NEUTRAL"
        except (KeyError, ValueError):
            return "NO_DATA"
    
    def find_atm_strike(self, options_df: pd.DataFrame, option_type: str = "call") -> pd.Series:
        """Find the at-the-money strike for given options."""
        type_options = options_df[options_df["type"] == option_type]
        if type_options.empty:
            return pd.Series(dtype=object)
        
        # Find strike closest to mean (approximation of ATM)
        mean_strike = type_options["strike"].mean()
        atm_idx = type_options["strike"].sub(mean_strike).abs().idxmin()
        return type_options.loc[atm_idx]
    
    def create_iron_condor_trade(self, day_options: pd.DataFrame, date: datetime, vix_level: float) -> Dict:
        """Create Iron Condor trade."""
        atm_call = self.find_atm_strike(day_options, "call")
        if atm_call.empty:
            return None
        
        atm_strike = atm_call["strike"]
        
        # Find corresponding ATM put
        atm_put = day_options[
            (day_options["type"] == "put") & 
            (day_options["strike"] == atm_strike)
        ]
        atm_put = atm_put.iloc[0] if not atm_put.empty else None
        
        # Find OTM options for wings
        wing_width = self.config.IRON_CONDOR_WING_WIDTH
        
        otm_call = day_options[
            (day_options["type"] == "call") & 
            (day_options["strike"] > atm_strike + wing_width)
        ].sort_values("strike").head(1)
        
        otm_put = day_options[
            (day_options["type"] == "put") & 
            (day_options["strike"] < atm_strike - wing_width)
        ].sort_values("strike", ascending=False).head(1)
        
        # Calculate P&L (simplified)
        credit = 0
        if not pd.isna(atm_call["bid"]):
            credit += atm_call["bid"]
        if atm_put is not None and not pd.isna(atm_put["bid"]):
            credit += atm_put["bid"]
        if not otm_call.empty and not pd.isna(otm_call["ask"].iloc[0]):
            credit -= otm_call["ask"].iloc[0]
        if not otm_put.empty and not pd.isna(otm_put["ask"].iloc[0]):
            credit -= otm_put["ask"].iloc[0]
        
        return {
            "date": date,
            "strategy": "IronCondor",
            "regime": "HIGH_VOL",
            "vix_level": vix_level,
            "short_call": atm_call["symbol"],
            "short_put": atm_put["symbol"] if atm_put is not None else None,
            "long_call": otm_call["symbol"].iloc[0] if not otm_call.empty else None,
            "long_put": otm_put["symbol"].iloc[0] if not otm_put.empty else None,
            "net_credit": credit,
            "position_size": self.config.POSITION_SIZE
        }
    
    def create_iron_butterfly_trade(self, day_options: pd.DataFrame, date: datetime, vix_level: float) -> Dict:
        """Create Iron Butterfly trade."""
        atm_call = self.find_atm_strike(day_options, "call")
        if atm_call.empty:
            return None
        
        atm_strike = atm_call["strike"]
        
        # Find corresponding ATM put
        atm_put = day_options[
            (day_options["type"] == "put") & 
            (day_options["strike"] == atm_strike)
        ]
        atm_put = atm_put.iloc[0] if not atm_put.empty else None
        
        # Find wings (further out for butterfly)
        wing_width = self.config.IRON_BUTTERFLY_WING_WIDTH
        
        wing_call = day_options[
            (day_options["type"] == "call") & 
            (day_options["strike"] > atm_strike + wing_width)
        ].sort_values("strike").head(1)
        
        wing_put = day_options[
            (day_options["type"] == "put") & 
            (day_options["strike"] < atm_strike - wing_width)
        ].sort_values("strike", ascending=False).head(1)
        
        # Calculate P&L
        credit = 0
        if not pd.isna(atm_call["bid"]):
            credit += atm_call["bid"]
        if atm_put is not None and not pd.isna(atm_put["bid"]):
            credit += atm_put["bid"]
        if not wing_call.empty and not pd.isna(wing_call["ask"].iloc[0]):
            credit -= wing_call["ask"].iloc[0]
        if not wing_put.empty and not pd.isna(wing_put["ask"].iloc[0]):
            credit -= wing_put["ask"].iloc[0]
        
        return {
            "date": date,
            "strategy": "IronButterfly",
            "regime": "HIGH_VOL",
            "vix_level": vix_level,
            "short_call": atm_call["symbol"],
            "short_put": atm_put["symbol"] if atm_put is not None else None,
            "long_call": wing_call["symbol"].iloc[0] if not wing_call.empty else None,
            "long_put": wing_put["symbol"].iloc[0] if not wing_put.empty else None,
            "net_credit": credit,
            "position_size": self.config.POSITION_SIZE
        }
    
    def create_diagonal_trade(self, day_options: pd.DataFrame, date: datetime, vix_level: float) -> Dict:
        """Create Diagonal spread trade."""
        atm_call = self.find_atm_strike(day_options, "call")
        if atm_call.empty:
            return None
        
        # For diagonal, we'd typically buy longer-dated and sell shorter-dated
        # Since we're dealing with 0DTE, this is simplified
        debit = atm_call["ask"] if not pd.isna(atm_call["ask"]) else 0
        
        return {
            "date": date,
            "strategy": "Diagonal",
            "regime": "LOW_VOL",
            "vix_level": vix_level,
            "long_call": atm_call["symbol"],
            "net_debit": debit,
            "position_size": self.config.POSITION_SIZE
        }
    
    def run_backtest(self, options_df: pd.DataFrame):
        """Run the complete backtest."""
        logger.info("Starting backtest...")
        
        # Load VIX data
        self.load_vix_data(options_df)
        
        # Generate business days
        business_days = pd.date_range(
            self.config.BACKTEST_START, 
            self.config.BACKTEST_END, 
            freq="B"
        )
        
        for date in business_days:
            regime = self.get_market_regime(date)
            
            if regime == "NO_DATA":
                continue
            
            # Get options for this day
            day_options = options_df[options_df["query_date"] == date]
            
            if day_options.empty:
                logger.warning(f"No options data for {date.date()}")
                continue
            
            vix_level = float(self.vix_data.loc[date]) if date in self.vix_data.index else np.nan
            
            # Execute trades based on regime
            if regime == "HIGH_VOL":
                # Iron Condor
                iron_condor = self.create_iron_condor_trade(day_options, date, vix_level)
                if iron_condor:
                    self.trade_log.append(iron_condor)
                
                # Iron Butterfly
                iron_butterfly = self.create_iron_butterfly_trade(day_options, date, vix_level)
                if iron_butterfly:
                    self.trade_log.append(iron_butterfly)
                    
            elif regime == "LOW_VOL":
                # Diagonal
                diagonal = self.create_diagonal_trade(day_options, date, vix_level)
                if diagonal:
                    self.trade_log.append(diagonal)
        
        logger.info(f"Backtest completed. Generated {len(self.trade_log)} trades.")
    
    def save_results(self):
        """Save backtest results to CSV files."""
        if not self.trade_log:
            logger.warning("No trades to save")
            return
        
        trade_df = pd.DataFrame(self.trade_log)
        trade_df.to_csv(self.config.TRADE_LOG_CSV, index=False)
        logger.info(f"Saved {len(trade_df)} trades to {self.config.TRADE_LOG_CSV}")
        
        # Basic performance analysis
        self.analyze_performance(trade_df)
    
    def analyze_performance(self, trade_df: pd.DataFrame):
        """Analyze backtest performance."""
        logger.info("Analyzing performance...")
        
        # Strategy counts
        strategy_counts = trade_df["strategy"].value_counts()
        logger.info(f"Strategy distribution:\n{strategy_counts}")
        
        # Regime distribution
        regime_counts = trade_df["regime"].value_counts()
        logger.info(f"Regime distribution:\n{regime_counts}")
        
        # Basic P&L analysis (simplified)
        if "net_credit" in trade_df.columns:
            total_credit = trade_df["net_credit"].sum()
            logger.info(f"Total net credit: ${total_credit:.2f}")
        
        if "net_debit" in trade_df.columns:
            total_debit = trade_df["net_debit"].sum()
            logger.info(f"Total net debit: ${total_debit:.2f}")


def main():
    """Main execution function."""
    config = BacktestConfig()
    
    logger.info("Starting 0DTE Options Strategy Backtest")
    logger.info(f"Period: {config.BACKTEST_START.date()} to {config.BACKTEST_END.date()}")
    logger.info(f"Underlying: {config.UNDERLYING}")
    
    # Initialize data fetcher
    data_fetcher = MultiSourceDataFetcher(config)
    
    # Fetch historical options data
    options_df = data_fetcher.fetch_historical_options_data()
    
    if options_df.empty:
        logger.error("No historical options data available. Exiting.")
        return
    
    # Save raw data
    options_df.to_csv(config.OPTIONS_CSV, index=False)
    logger.info(f"Saved {len(options_df)} option records to {config.OPTIONS_CSV}")
    
    # Run backtest
    strategy = ZeroDTEStrategy(config)
    strategy.run_backtest(options_df)
    
    # Save results
    strategy.save_results()
    
    logger.info("Backtest completed successfully!")


if __name__ == "__main__":
    main() 