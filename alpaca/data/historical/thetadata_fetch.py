import requests

base_paths = [
    '',
    '/api',
    '/rest',
    '/option',
    '/option/list',
]

for base in base_paths:
    if base.endswith('/'):
        url = f"http://localhost:25510{base}roots"
    else:
        url = f"http://localhost:25510{base}/roots"
    print(f"Testing: {url}")
    try:
        response = requests.get(url)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}\n---\n")
    except Exception as e:
        print(f"Error: {e}\n---\n") 