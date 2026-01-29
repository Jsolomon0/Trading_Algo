# Copilot Instructions Improvements

## 1. Neural Network Architecture
```
[Input] BTC/USD OHLCV Data
    ↓
[Feature Engineering] get_ichimoku_features()
    ↓                     
[Sequence Formation] (30 timesteps, 6 features)
    ↓
[LSTM Backbone]
    Input Size: 6
    Hidden Size: 128
    Layers: 2
    Dropout: 0.2
    ↓
[Actor Head]             [Critic Head]
Linear(128→64)          Linear(128→64)
ReLU + Dropout(0.2)     ReLU + Dropout(0.2)
Linear(64→3)            Linear(64→1)
Softmax                 (Value estimate)
(Action probabilities)
```

## 2. Data Flow Pipeline

### Training Flow
```
1. Data Ingestion:
   raw_data = fetch_training_data(days=30)
   → CryptoHistoricalDataClient → Minute bars

2. Feature Engineering:
   features_df = prepare_features(raw_data)
   → get_ichimoku_features() → 6 technical indicators

3. Training Loop:
   feature_sequence = deque(maxlen=30)
   → agent.select_action(state_sequence)
   → environment.calculate_reward(action, price)
   → agent.train_step(batch)
```

### Live Trading Flow
```
1. Historical Warmup:
   await fetch_historical_data()
   → Build initial feature sequence

2. Live Updates:
   on_minute_bar_update(bar)
   → Update feature sequence
   → Generate features
   → Model inference
   → Order execution
```

## 3. Environment Validation Script

```python
def validate_environment():
    """Validate the Python environment and dependencies."""
    import sys
    import torch
    import pandas as pd
    from alpaca.data import CryptoHistoricalDataClient
    
    # Check Python version
    print(f"Python version: {sys.version}")
    assert sys.version_info >= (3, 8), "Python 3.8+ required"
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Verify Alpaca connection
    try:
        client = CryptoHistoricalDataClient(
            api_key=os.getenv('ALPACA_API_KEY_ID'),
            secret_key=os.getenv('ALPACA_API_SECRET_KEY')
        )
        print("✓ Alpaca API connection successful")
    except Exception as e:
        print(f"✗ Alpaca API error: {str(e)}")
    
    # List package versions
    packages = [pd, np, torch, matplotlib]
    for pkg in packages:
        print(f"{pkg.__name__}: {pkg.__version__}")
```

## 4. Common Issues & Solutions

### Feature Calculation
- **Issue**: NaN values in Ichimoku features
  - **Cause**: Insufficient historical data for lookback periods
  - **Solution**: Ensure raw data length > max(52, 26) periods

- **Issue**: Unexpected feature scaling
  - **Cause**: Missing normalization in `get_ichimoku_features`
  - **Solution**: Check `feature_normalized_distance_from_cloud` calculation

### Model Training
- **Issue**: Loss explosion/NaN
  - **Cause**: Learning rate too high or gradient clipping missing
  - **Solution**: Add `torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)`

- **Issue**: Poor convergence
  - **Cause**: Insufficient replay buffer samples
  - **Solution**: Wait for buffer.size() >= 32 before training

### Live Trading
- **Issue**: Stale feature sequence
  - **Cause**: Missed market data updates
  - **Solution**: Implement heartbeat monitoring in `on_minute_bar_update`

## 5. Performance Tuning Guide

### Batch Size Impact
- Default: 32
- Increase if GPU memory allows
- Monitor training stability
- Suggested range: 32-128

### Sequence Length Analysis
- Current: 30 timesteps
- Tradeoffs:
  - Longer = More context but slower training
  - Shorter = Faster but may miss patterns
- Test range: 20-50 timesteps

### Feature Normalization
Critical factors:
- Cloud distance normalization (relative to price)
- Cross signals (binary features)
- Position indicators (-1, 0, 1 scale)

## 6. Test Framework Examples

### Feature Calculation Tests
```python
def test_ichimoku_features():
    # Test case: Simple uptrend
    df = pd.DataFrame({
        'open': [100] * 60,
        'high': [110] * 60,
        'low': [90] * 60,
        'close': list(range(100, 160)),
        'volume': [1000] * 60
    })
    
    features = get_ichimoku_features(df)
    assert features['feature_price_cloud_position'].iloc[-1] == 1
    assert features['feature_tk_cross'].iloc[-1] == 1
```

### Model Validation Tests
```python
def test_model_deterministic():
    agent = DRLTradingAgent()
    state = torch.randn(1, 30, 6)
    
    # Test deterministic predictions
    action1, _, _ = agent.select_action(state, deterministic=True)
    action2, _, _ = agent.select_action(state, deterministic=True)
    assert action1 == action2
```

### Backtest Validation
```python
def validate_backtest_results(results):
    # No impossible returns
    assert abs(results['returns']).max() < 0.5
    
    # Transaction costs applied
    trades = results[results['action'] != 'HOLD']
    assert (trades['returns'] <= -0.001).all()
    
    # No trading gaps
    gaps = trades.index.to_series().diff()
    assert gaps.max() <= pd.Timedelta('2min')
```

## Future Improvements

1. **Model Versioning**:
   - Add model version tracking in saved artifacts
   - Include training configuration in saves
   - Implement backwards compatibility

2. **Risk Management**:
   - Add position sizing logic
   - Implement stop-loss mechanisms
   - Add volatility-based filters

3. **Online Learning**:
   - Enable continuous model updates
   - Implement experience prioritization
   - Add catastrophic forgetting prevention