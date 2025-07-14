#!/usr/bin/env python3
"""
Setup script for historical options data fetching environment.

This script will help you:
1. Install required dependencies
2. Verify ThetaData Terminal setup
3. Test API connections
4. Run the improved backtest
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def check_conda_environment():
    """Check if conda environment is active."""
    print("\n" + "=" * 50)
    print("Checking Conda Environment")
    print("=" * 50)
    
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"✅ Conda environment active: {conda_env}")
        return True
    else:
        print("❌ No conda environment detected")
        print("Please activate a conda environment before running this script")
        print("Example: conda activate your_env_name")
        return False

def install_dependencies():
    """Install required Python packages."""
    print("\n" + "=" * 50)
    print("Installing Dependencies")
    print("=" * 50)
    
    # Required packages
    packages = [
        "pandas",
        "numpy",
        "httpx",
        "yfinance",
        "python-dotenv",
        "thetadata"  # If available
    ]
    
    for package in packages:
        print(f"\nInstalling {package}...")
        success = run_command(f"pip install {package}", f"Installing {package}")
        if not success and package != "thetadata":  # thetadata might not be available
            print(f"Failed to install {package}. Please install manually.")
            return False
    
    print("\n✅ All dependencies installed successfully!")
    return True

def setup_environment_file():
    """Create or check .env file for API keys."""
    print("\n" + "=" * 50)
    print("Setting up Environment File")
    print("=" * 50)
    
    env_file = Path(".env")
    
    if env_file.exists():
        print(f"✅ Found existing .env file: {env_file.absolute()}")
        with open(env_file, 'r') as f:
            content = f.read()
            if "APCA_API_KEY_ID" in content:
                print("✅ Alpaca API keys found in .env file")
            else:
                print("❌ Alpaca API keys not found in .env file")
    else:
        print("Creating .env file template...")
        env_template = """# Alpaca API Keys
APCA_API_KEY_ID=your_alpaca_api_key_here
APCA_API_SECRET_KEY=your_alpaca_secret_key_here

# Optional: ThetaData API Key (if using cloud version)
THETA_API_KEY=your_theta_api_key_here
"""
        with open(env_file, 'w') as f:
            f.write(env_template)
        print(f"✅ Created .env template: {env_file.absolute()}")
        print("Please edit the .env file and add your API keys")
    
    return True

def check_thetadata_terminal():
    """Check if ThetaData Terminal is running."""
    print("\n" + "=" * 50)
    print("Checking ThetaData Terminal")
    print("=" * 50)
    
    try:
        import httpx
        response = httpx.get("http://127.0.0.1:25510/v2/option/list/roots", timeout=5)
        if response.status_code == 200:
            print("✅ ThetaData Terminal is running and accessible")
            return True
        else:
            print(f"❌ ThetaData Terminal responded with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ ThetaData Terminal not accessible: {e}")
        print("\nTo start ThetaData Terminal:")
        print("1. Make sure you have ThetaData Terminal installed")
        print("2. Run: java -jar ThetaTerminal.jar")
        print("3. Wait for it to start on port 25510")
        return False

def run_test_connection():
    """Run the connection test script."""
    print("\n" + "=" * 50)
    print("Testing Connections")
    print("=" * 50)
    
    test_script = Path("test_connection.py")
    if test_script.exists():
        print("Running connection test...")
        success = run_command(f"python {test_script}", "Connection test")
        return success
    else:
        print(f"❌ Test script not found: {test_script}")
        return False

def run_backtest():
    """Run the improved backtest."""
    print("\n" + "=" * 50)
    print("Running Backtest")
    print("=" * 50)
    
    backtest_script = Path("improved_zero_dte_backtest.py")
    if backtest_script.exists():
        print("Starting backtest...")
        success = run_command(f"python {backtest_script}", "Backtest execution")
        return success
    else:
        print(f"❌ Backtest script not found: {backtest_script}")
        return False

def main():
    """Main setup function."""
    print("Historical Options Data Fetching Environment Setup")
    print("=" * 60)
    
    # Check conda environment
    if not check_conda_environment():
        return
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies. Please resolve and try again.")
        return
    
    # Setup environment file
    setup_environment_file()
    
    # Check ThetaData Terminal
    theta_running = check_thetadata_terminal()
    
    # Run connection test
    if theta_running:
        test_success = run_test_connection()
        if test_success:
            print("\n✅ All tests passed! Ready to run backtest.")
            
            # Ask if user wants to run backtest
            response = input("\nWould you like to run the backtest now? (y/n): ")
            if response.lower() in ['y', 'yes']:
                run_backtest()
            else:
                print("\nTo run the backtest later, use:")
                print("python improved_zero_dte_backtest.py")
        else:
            print("\n❌ Connection tests failed. Please check ThetaData Terminal and API keys.")
    else:
        print("\n❌ ThetaData Terminal not running. Please start it and try again.")
        print("\nNext steps:")
        print("1. Start ThetaData Terminal: java -jar ThetaTerminal.jar")
        print("2. Run connection test: python test_connection.py")
        print("3. Run backtest: python improved_zero_dte_backtest.py")

if __name__ == "__main__":
    main() 