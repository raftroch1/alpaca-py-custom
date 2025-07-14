import os
from datetime import datetime
import pandas as pd
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest, OptionBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

def load_api_keys():
    load_dotenv()
    return os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY")

def get_option_chain_snapshot(client, underlying_symbol, as_of_date):
    request = OptionChainRequest(
        underlying_symbol=underlying_symbol,
        updated_since=as_of_date
    )
    return client.get_option_chain(request)

def get_option_bars(client, contract_symbol, start, end):
    request = OptionBarsRequest(
        symbol_or_symbols=contract_symbol,
        start=start,
        end=end,
        timeframe=TimeFrame(amount=1, unit=TimeFrameUnit.Day)
    )
    bars = client.get_option_bars(request)
    return bars.df if hasattr(bars, "df") else pd.DataFrame()

def fetch_historical_option_chain(underlying_symbol, as_of_date, bar_start, bar_end, max_contracts=10):
    api_key, api_secret = load_api_keys()
    client = OptionHistoricalDataClient(api_key=api_key, secret_key=api_secret)
    chain = get_option_chain_snapshot(client, underlying_symbol, as_of_date)
    contract_symbols = list(chain.keys())[:max_contracts]  # Limit for demo/backtest speed

    all_bars = []
    for symbol in contract_symbols:
        df = get_option_bars(client, symbol, bar_start, bar_end)
        if not df.empty:
            df["contract_symbol"] = symbol
            all_bars.append(df)
    if all_bars:
        return pd.concat(all_bars, ignore_index=True)
    else:
        return pd.DataFrame()

# Example usage:
if __name__ == "__main__":
    underlying = "AAPL"
    as_of = datetime(2024, 4, 25)
    bar_start = datetime(2024, 4, 20)
    bar_end = datetime(2024, 4, 25)
    df = fetch_historical_option_chain(underlying, as_of, bar_start, bar_end)
    print(df.head())
    # Now df is ready to be fed into your backtest library! 