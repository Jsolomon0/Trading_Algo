import os
import torch
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from alpaca.data import CryptoHistoricalDataClient

def validate_setup():
    """Validate the environment setup and dependencies."""
    print("\n=== Environment Validation ===")
    
    # Check Python packages
    print("\nPackage versions:")
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Test Alpaca connection
    print("\nTesting Alpaca connection:")
    load_dotenv()
    api_key = os.getenv('ALPACA_API_KEY_ID')
    secret_key = os.getenv('ALPACA_API_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ Missing Alpaca API credentials in .env file")
        return False
    
    try:
        client = CryptoHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        print("✓ Successfully connected to Alpaca API")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Alpaca: {str(e)}")
        return False

if __name__ == "__main__":
    validate_setup()