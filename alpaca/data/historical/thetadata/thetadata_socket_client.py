"""
ThetaData Socket Client for Historical Options Data

This client uses the ThetaData Terminal's socket API (CLIENT_PORT=11000)
instead of HTTP REST API to fetch historical options data.

Based on ThetaData Terminal configuration:
- CLIENT_PORT=11000 - Python API query-based (MDDS) socket
- STREAM_PORT=10000 - Python API streaming (FPSS)
- WS_PORT=25520 - WebSocket server
- HTTP_PORT=25510 - HTTP server (not for REST API)
"""

import socket
import json
import struct
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import time
import logging

try:
    import pandas as pd
    DataFrame = pd.DataFrame
except ImportError:
    pd = None
    DataFrame = None

logger = logging.getLogger(__name__)

class ThetaDataSocketClient:
    """
    Socket-based client for ThetaData Terminal.
    
    This client connects to ThetaData Terminal using socket connections
    on the configured ports (CLIENT_PORT=11000 for queries).
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 11000, timeout: int = 30):
        """
        Initialize ThetaData socket client.
        
        Args:
            host: ThetaData Terminal host (default: 127.0.0.1)
            port: ThetaData Terminal CLIENT_PORT (default: 11000)
            timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None
        self.connected = False
        
    def connect(self) -> bool:
        """
        Connect to ThetaData Terminal.
        
        Returns:
            True if connected successfully, False otherwise
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.connected = True
            logger.info(f"Connected to ThetaData Terminal at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to ThetaData Terminal: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from ThetaData Terminal."""
        if self.socket:
            self.socket.close()
            self.connected = False
            logger.info("Disconnected from ThetaData Terminal")
    
    def send_message(self, message: bytes) -> Optional[bytes]:
        """
        Send a message to ThetaData Terminal and receive response.
        
        Args:
            message: Message to send as bytes
            
        Returns:
            Response bytes or None if error
        """
        if not self.connected:
            logger.error("Not connected to ThetaData Terminal")
            return None
            
        try:
            # Send message
            if self.socket is not None:
                self.socket.send(message)
                
                # Receive response
                response = self.socket.recv(4096)
                return response
            return None
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return None
    
    def send_command(self, command: str) -> Optional[str]:
        """
        Send a command string to ThetaData Terminal.
        
        Args:
            command: Command string to send
            
        Returns:
            Response string or None if error
        """
        try:
            # Convert command to bytes
            message = command.encode('utf-8')
            
            # Send and receive response
            response_bytes = self.send_message(message)
            if response_bytes:
                return response_bytes.decode('utf-8')
            return None
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return None
    
    def get_option_chain(self, symbol: str, exp_date: str) -> Optional[pd.DataFrame]:
        """
        Get options chain for a symbol and expiration date.
        
        Args:
            symbol: Underlying symbol (e.g., 'SPY')
            exp_date: Expiration date in YYYY-MM-DD format
            
        Returns:
            DataFrame with options chain data or None if error
        """
        try:
            # Format command for options chain
            command = f"GET_OPTION_CHAIN:{symbol}:{exp_date}"
            
            response = self.send_command(command)
            if response:
                # Parse response and convert to DataFrame
                # This is a placeholder - actual parsing depends on ThetaData's response format
                logger.info(f"Options chain response: {response}")
                return pd.DataFrame()  # Placeholder
            return None
        except Exception as e:
            logger.error(f"Error getting options chain: {e}")
            return None
    
    def get_historical_quotes(self, contract: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Get historical quotes for an options contract.
        
        Args:
            contract: Options contract symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with historical quotes or None if error
        """
        try:
            command = f"GET_HISTORICAL_QUOTES:{contract}:{start_date}:{end_date}"
            
            response = self.send_command(command)
            if response:
                logger.info(f"Historical quotes response: {response}")
                return pd.DataFrame()  # Placeholder
            return None
        except Exception as e:
            logger.error(f"Error getting historical quotes: {e}")
            return None
    
    def test_connection(self) -> bool:
        """
        Test connection to ThetaData Terminal.
        
        Returns:
            True if connection test successful, False otherwise
        """
        try:
            # Try to send a simple test command
            response = self.send_command("TEST")
            if response:
                logger.info(f"Connection test response: {response}")
                return True
            return False
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


def test_theta_socket_connection():
    """Test the ThetaData socket connection."""
    print("Testing ThetaData Socket Connection")
    print("=" * 50)
    
    client = ThetaDataSocketClient()
    
    # Test connection
    if client.connect():
        print("✅ Connected to ThetaData Terminal")
        
        # Test basic communication
        if client.test_connection():
            print("✅ Connection test passed")
        else:
            print("❌ Connection test failed")
            
        # Test options chain request
        print("\n🔍 Testing options chain request...")
        df = client.get_option_chain("SPY", "2024-06-14")
        if df is not None:
            print("✅ Options chain request successful")
        else:
            print("❌ Options chain request failed")
            
        client.disconnect()
    else:
        print("❌ Failed to connect to ThetaData Terminal")
        print("Make sure ThetaData Terminal is running and CLIENT_PORT=11000 is configured")


if __name__ == "__main__":
    test_theta_socket_connection() 