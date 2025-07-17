#!/bin/bash
# ThetaData Terminal Restart Script
# This script helps restart ThetaData Terminal with proper connection to MDDS servers

echo "🔄 Restarting ThetaData Terminal..."

# Kill existing ThetaTerminal processes
echo "1️⃣  Stopping existing ThetaTerminal processes..."
pkill -f "ThetaTerminal.jar" 2>/dev/null
sleep 2

# Navigate to ThetaData directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if ThetaTerminal.jar exists
if [ ! -f "ThetaTerminal.jar" ]; then
    echo "❌ ThetaTerminal.jar not found in current directory!"
    echo "   Expected location: $SCRIPT_DIR/ThetaTerminal.jar"
    exit 1
fi

echo "2️⃣  Starting ThetaTerminal.jar..."
echo "   📍 Working directory: $SCRIPT_DIR"
echo "   ⚠️  You'll need to enter your ThetaData credentials when prompted"
echo ""

# Start ThetaTerminal (user will need to provide credentials)
echo "🚀 Launching ThetaData Terminal..."
echo "   Please enter your credentials when prompted:"
echo "   - Email: [your ThetaData email]"
echo "   - Password: [your ThetaData password]"
echo ""

java -jar ThetaTerminal.jar

echo ""
echo "✅ ThetaTerminal.jar started"
echo "⏳ Wait 30-60 seconds for connection to establish"
echo "🔍 Test connection with: python -c 'from connector import ThetaDataConnector; ThetaDataConnector().test_connection()'" 