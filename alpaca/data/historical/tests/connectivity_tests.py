"""
Comprehensive connectivity tests for historical data sources.
"""

import requests
import socket
import json
from datetime import datetime, timedelta


class ConnectivityTester:
    """Test connectivity to various data sources."""
    
    def __init__(self):
        self.thetadata_rest_port = 25510
        self.thetadata_socket_port = 11000
        self.thetadata_host = "127.0.0.1"
        self.results = {}
    
    def test_thetadata_terminal_running(self) -> bool:
        """Test if ThetaData Terminal is running and accessible."""
        try:
            response = requests.get(
                f"http://{self.thetadata_host}:{self.thetadata_rest_port}/v2/list/roots/option",
                timeout=5
            )
            rest_ok = response.status_code == 200
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            socket_ok = sock.connect_ex((self.thetadata_host, self.thetadata_socket_port)) == 0
            sock.close()
            
            return rest_ok and socket_ok
        except Exception:
            return False
    
    def run_all_tests(self):
        """Run all connectivity tests."""
        print("🔍 Running connectivity tests...")
        tests = [
            ("ThetaData Terminal", self.test_thetadata_terminal_running),
        ]
        
        for test_name, test_func in tests:
            print(f"Testing {test_name}...", end=" ")
            success = test_func()
            print("✅ PASS" if success else "❌ FAIL")


if __name__ == "__main__":
    tester = ConnectivityTester()
    tester.run_all_tests()
