import os
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest

# --- CONFIG ---
UNDERLYING = "SPY"
BACKTEST_START = datetime(2025, 4, 13)
BACKTEST_END = datetime(2025, 7, 13)
VIX_LOW = 17
VIX_HIGH = 18

# --- LOAD API KEYS ---
load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
client = OptionHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)

# --- FETCH VIX DATA ---
vix = yf.download("^VIX", start=BACKTEST_START, end=BACKTEST_END)
vix_series = vix["Close"].reindex(pd.date_range(BACKTEST_START, BACKTEST_END, freq="B")).ffill()

# --- BACKTEST LOOP ---
results = []
for day in pd.date_range(BACKTEST_START, BACKTEST_END, freq="B"):
    vix_today = vix_series.loc[day] if day in vix_series else None
    if vix_today is None:
        continue
    # Determine regime
    if vix_today > VIX_HIGH:
        regime = "HIGH_VOL"
    elif vix_today < VIX_LOW:
        regime = "LOW_VOL"
    else:
        regime = "NO_TRADE"
    # Fetch 0DTE option chain for the day
    chain_req = OptionChainRequest(
        underlying_symbol=UNDERLYING,
        updated_since=day
    )
    chain = client.get_option_chain(chain_req)
    contracts = []
    for symbol, data in chain.items():
        exp_date = getattr(data, 'expiration_date', None)
        if exp_date and str(exp_date) == day.strftime("%Y-%m-%d"):
            contracts.append({
                "symbol": symbol,
                "type": getattr(data, 'type', None),
                "strike": getattr(data, 'strike_price', None),
                "expiration": exp_date
            })
    contracts_df = pd.DataFrame(contracts)
    print(f"{day.date()} | Regime: {regime} | 0DTE contracts: {len(contracts_df)}")
    results.append({
        "date": day,
        "regime": regime,
        "contracts": contracts_df
    })

# --- SAVE OR FURTHER PROCESSING ---
# Example: Save summary of contract counts per day
summary = pd.DataFrame({
    "date": [r["date"] for r in results],
    "regime": [r["regime"] for r in results],
    "num_contracts": [len(r["contracts"]) for r in results]
})
summary.to_csv("zero_dte_backtest_summary.csv", index=False)
print("Backtest summary saved to zero_dte_backtest_summary.csv") 