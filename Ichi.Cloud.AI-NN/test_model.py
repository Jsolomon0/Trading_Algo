import pandas as pd
import numpy as np
import torch
from trainer0 import get_ichimoku_features
from drl_agent import DRLTradingAgent
from collections import deque

def test_feature_engineering():
    """Test the Ichimoku feature engineering function."""
    print("\n=== Testing Feature Engineering ===")
    
    # Create sample data
    df = pd.DataFrame({
        'open': [100] * 60,
        'high': [110] * 60,
        'low': [90] * 60,
        'close': list(range(100, 160)),
        'volume': [1000] * 60
    })
    
    try:
        features = get_ichimoku_features(df)
        print(f"✓ Features generated successfully")
        print(f"Feature shape: {features.shape}")
        print("\nFeature columns:")
        for col in features.columns:
            print(f"- {col}")
        return features
    except Exception as e:
        print(f"❌ Feature generation failed: {str(e)}")
        return None

def test_model_loading(features_df):
    """Test model loading and inference."""
    print("\n=== Testing Model Loading ===")
    
    try:
        # Initialize agent with CPU device
        device = torch.device('cpu')
        agent = DRLTradingAgent(device=device)
        print("✓ Agent initialized successfully")
        
        # Load saved model
        try:
            agent.load_model('drl_trading_model.pth')
            print("✓ Loaded saved model")
        except:
            print("ℹ No saved model found, using initialized model")
        
        # Prepare feature sequence like in training
        feature_matrix = features_df.values
        feature_sequence = deque(maxlen=30)
        
        # Initialize sequence with first 30 timesteps
        for i in range(30):
            feature_sequence.append(feature_matrix[i])
        
        # Convert to tensor
        state = torch.FloatTensor(np.array(list(feature_sequence))).unsqueeze(0)
        state = state.to(device)
        
        # Test inference
        action, name, prob = agent.select_action(state, deterministic=True)
        print(f"✓ Model inference successful")
        print(f"Action: {name} (probability: {prob:.4f})")
        return True
    except Exception as e:
        print(f"❌ Model testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run feature engineering test
    features_df = test_feature_engineering()
    
    # Run model test if features were generated successfully
    if features_df is not None:
        test_model_loading(features_df)