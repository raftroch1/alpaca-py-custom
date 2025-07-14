from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest
from datetime import datetime
import os
from dotenv import load_dotenv

# Load API keys
load_dotenv()
API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

# 1. Fetch the historical option chain for SPY as of 2025-04-14
client = OptionHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)
as_of_date = datetime(2025, 4, 14)  # The historical date you want the chain for

request = OptionChainRequest(
    underlying_symbol="SPY",
    updated_since=as_of_date
)
chain = client.get_option_chain(request)

# Print the number of contracts and the first 10 contract symbols found
if hasattr(chain, 'option_contracts'):
    contracts = chain.option_contracts
    print(f"Contracts found: {len(contracts)}")
    for i, contract in enumerate(contracts[:10]):
        print(f"{i+1}: {contract.symbol}")
    if not contracts:
        print(f"No contracts found in the historical option chain for SPY as of {as_of_date.date()}.")
else:
    print(f"No contracts attribute in chain response: {chain}") 