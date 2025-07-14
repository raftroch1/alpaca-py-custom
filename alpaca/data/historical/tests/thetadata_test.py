import httpx

BASE_URL = "http://127.0.0.1:25510/v2"
params = {'root': 'SPY'}

url = BASE_URL + '/option/list/expirations'
response = httpx.get(url, params=params, timeout=60)
print("Status code:", response.status_code)
print("Response:", response.text[:1000])  # Print first 1000 chars 