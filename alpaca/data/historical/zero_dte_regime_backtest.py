import os
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, OptionLatestQuoteRequest
import time
import re

# --- CONFIG ---
UNDERLYING = "SPY"
BACKTEST_START = datetime(2024, 6, 13)  # Fixed: Changed from 2025 to 2024
BACKTEST_END = datetime(2024, 7, 13)    # Fixed: Changed from 2025 to 2024
OPTIONS_CSV = "spy_options_3mo.csv"
TRADE_LOG_CSV = "options_regime_trades.csv"
VIX_LOW = 17
VIX_HIGH = 18

# --- LOAD API KEYS ---
load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
client = OptionHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)

def parse_occ_expiration(symbol):
    """Parse expiration date from OCC option symbol (e.g., SPY250717C00603000 -> 2025-07-17)."""
    # OCC format: SYMBOLYYMMDDTSSSSSSS
    m = re.match(r"^[A-Z]+(\d{2})(\d{2})(\d{2})[CP]", symbol)
    if not m:
        return None
    year, month, day = m.groups()
    year = int(year)
    # Assume 2000s for 00-99
    year += 2000
    return datetime(year, int(month), int(day)).date()

# --- FETCH HISTORICAL OPTION CHAINS & QUOTES ---
all_rows = []
business_days = pd.date_range(BACKTEST_START, BACKTEST_END, freq="B")
for query_date in business_days:
    print(f"Fetching option chain for {query_date.date()}...")
    try:
        chain_req = OptionChainRequest(underlying_symbol=UNDERLYING, updated_since=query_date)
        chain = client.get_option_chain(chain_req)
        # Fix: chain is a dict of symbol -> contract data
        if isinstance(chain, dict):
            contracts = list(chain.values())
        else:
            contracts = []
        print(f"  Contracts found: {len(contracts)}")
        if contracts:
            print("   First 3 contract symbols:", [getattr(c, 'symbol', None) for c in contracts[:3]])
            # Debug: print parsed expiration dates for first 10 contracts
            for c in contracts[:10]:
                symbol = getattr(c, 'symbol', None)
                exp_parsed = parse_occ_expiration(symbol) if symbol else None
                print(f"    Contract: {symbol}, Parsed Exp: {exp_parsed}, Query: {query_date.date()}")
        # Robust 0DTE filter
        zero_dte_contracts = []
        for c in contracts:
            symbol = getattr(c, 'symbol', None)
            exp_parsed = parse_occ_expiration(symbol) if symbol else None
            if exp_parsed == query_date.date():
                zero_dte_contracts.append(c)
        print(f"    0DTE contracts found: {len(zero_dte_contracts)}")
        if not zero_dte_contracts:
            continue
        # Fetch quotes for all contracts (batch if possible)
        for contract in zero_dte_contracts:
            symbol = getattr(contract, 'symbol', None)
            if symbol is None:
                print(f"    Skipping contract with missing symbol on {query_date.date()}")
                continue
            try:
                # OptionLatestQuoteRequest does not support 'as_of', so this fetches the latest available quote (not historical)
                quote_req = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
                quote = client.get_option_latest_quote(quote_req)
                q = quote[symbol]
                row = {
                    "query_date": query_date,
                    "expiration": getattr(contract, 'expiration_date', None),
                    "type": (getattr(contract, 'type', '') or '').lower(),
                    "strike": getattr(contract, 'strike_price', None),
                    "bid": getattr(q, 'bid', None),
                    "ask": getattr(q, 'ask', None),
                    "symbol": symbol,
                    "last_quote_time": getattr(q, 'timestamp', None)
                }
                all_rows.append(row)
            except Exception as e:
                print(f"    Quote fetch failed for {symbol} on {query_date.date()}: {e}")
            time.sleep(0.1)  # avoid rate limits
    except Exception as e:
        print(f"  Chain fetch failed for {query_date.date()}: {e}")
    time.sleep(0.2)  # avoid rate limits

if not all_rows:
    print("No option contracts were found for any day in the backtest window.")

options_df = pd.DataFrame(all_rows)
options_df.to_csv(OPTIONS_CSV, index=False)
print(f"Saved {len(options_df)} option rows to {OPTIONS_CSV}")

# --- LOAD VIX DATA ---
if not options_df.empty:
    start_date = options_df["query_date"].min()
    end_date = options_df["query_date"].max()
    vix = yf.download("^VIX", start=start_date, end=end_date)
    vix_series = vix["Close"].reindex(pd.date_range(start_date, end_date, freq="B")).ffill()
else:
    vix_series = pd.Series(dtype=float)

# --- STRATEGY LOGIC ---
trade_log = []
for day in pd.date_range(BACKTEST_START, BACKTEST_END, freq="B"):
    try:
        vix_today = float(vix_series.loc[day])
    except Exception:
        continue  # Skip if VIX data is missing for this day
    regime = "NO_TRADE"
    if vix_today > VIX_HIGH:
        regime = "HIGH_VOL"
    elif vix_today < VIX_LOW:
        regime = "LOW_VOL"
    # Filter options for this day and 0DTE
    day_opts = options_df[options_df["query_date"] == day]
    if day_opts.empty or regime == "NO_TRADE":
        continue
    # --- HIGH VOL: Sell Iron Condor & Iron Butterfly ---
    if regime == "HIGH_VOL":
        # Iron Condor: Sell ATM call/put, buy further OTM call/put
        atm_strike = day_opts.loc[(day_opts["type"] == "call"), "strike"].sub(day_opts["strike"].mean()).abs().idxmin()
        atm_call = day_opts.loc[atm_strike]
        atm_put = day_opts.loc[(day_opts["type"] == "put") & (day_opts["strike"] == atm_call["strike"])]
        if not atm_put.empty:
            atm_put = atm_put.iloc[0]
        else:
            atm_put = None
        # OTM call/put (5 strikes away)
        otm_call = day_opts[(day_opts["type"] == "call") & (day_opts["strike"] > atm_call["strike"] + 5)].sort_values("strike").head(1)
        otm_put = day_opts[(day_opts["type"] == "put") & (day_opts["strike"] < atm_call["strike"] - 5)].sort_values("strike", ascending=False).head(1)
        # Iron Condor trade log
        trade_log.append({
            "date": day,
            "regime": regime,
            "strategy": "IronCondor",
            "short_call": atm_call["symbol"] if atm_call is not None else None,
            "short_put": atm_put["symbol"] if atm_put is not None else None,
            "long_call": otm_call["symbol"].values[0] if not otm_call.empty else None,
            "long_put": otm_put["symbol"].values[0] if not otm_put.empty else None,
            "short_call_bid": atm_call["bid"] if atm_call is not None else None,
            "short_put_bid": atm_put["bid"] if atm_put is not None else None,
            "long_call_ask": otm_call["ask"].values[0] if not otm_call.empty else None,
            "long_put_ask": otm_put["ask"].values[0] if not otm_put.empty else None,
        })
        # Iron Butterfly: Sell ATM straddle, buy OTM wings
        otm_call_bfly = day_opts[(day_opts["type"] == "call") & (day_opts["strike"] > atm_call["strike"] + 10)].sort_values("strike").head(1)
        otm_put_bfly = day_opts[(day_opts["type"] == "put") & (day_opts["strike"] < atm_call["strike"] - 10)].sort_values("strike", ascending=False).head(1)
        trade_log.append({
            "date": day,
            "regime": regime,
            "strategy": "IronButterfly",
            "short_call": atm_call["symbol"] if atm_call is not None else None,
            "short_put": atm_put["symbol"] if atm_put is not None else None,
            "long_call": otm_call_bfly["symbol"].values[0] if not otm_call_bfly.empty else None,
            "long_put": otm_put_bfly["symbol"].values[0] if not otm_put_bfly.empty else None,
            "short_call_bid": atm_call["bid"] if atm_call is not None else None,
            "short_put_bid": atm_put["bid"] if atm_put is not None else None,
            "long_call_ask": otm_call_bfly["ask"].values[0] if not otm_call_bfly.empty else None,
            "long_put_ask": otm_put_bfly["ask"].values[0] if not otm_put_bfly.empty else None,
        })
    # --- LOW VOL: Buy Diagonal ---
    elif regime == "LOW_VOL":
        atm_strike = day_opts.loc[(day_opts["type"] == "call"), "strike"].sub(day_opts["strike"].mean()).abs().idxmin()
        atm_call = day_opts.loc[atm_strike]
        trade_log.append({
            "date": day,
            "regime": regime,
            "strategy": "Diagonal",
            "long_call": atm_call["symbol"],
            "long_call_ask": atm_call["ask"],
        })

trade_log_df = pd.DataFrame(trade_log)
trade_log_df.to_csv(TRADE_LOG_CSV, index=False)
print(f"Saved {len(trade_log_df)} trades to {TRADE_LOG_CSV}")
print(trade_log_df["strategy"].value_counts())
print(trade_log_df.head()) 