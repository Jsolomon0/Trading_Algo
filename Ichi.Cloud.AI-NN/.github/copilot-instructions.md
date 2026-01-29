## Purpose

Short, actionable notes for AI coding agents working on this DRL trading repo. Focus on the minimal context needed to be immediately productive: architecture, key files, conventions, workflows, and concrete examples.

## Big picture (what this repo does)
- Implements an LSTM-based Actor-Critic DRL trading agent trained with a simplified PPO flow
- Features are 6 Ichimoku-derived indicators (sequence length 30) used to predict Buy/Sell/Hold
- Training and backtesting use historical BTC/USD data via Alpaca's crypto API
- Model saves to `drl_trading_model.pth` for live inference/trading

## Architecture Overview
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

### Data Pipeline
```
Training Flow:
1. fetch_training_data() → CryptoHistoricalDataClient → Minute bars
2. prepare_features() → get_ichimoku_features() → 6 indicators
3. Training loop → select_action → calculate_reward → train_step

Live Trading Flow:
1. fetch_historical_data() → Build initial sequence
2. on_minute_bar_update() → Update features → Model inference
```

## Key files (entry points & patterns)
- `drl_agent.py` — Core implementation:
  - `LSTMActorCritic` (LSTM backbone + actor/critic heads)
  - `DRLTradingAgent` (select_action, train_step, save/load)
  - `ReplayBuffer` and `TradingEnvironment` (reward calc & episode handling)
- `train_backtest.py` — Training pipeline:
  - Data fetch via Alpaca crypto client 
  - Feature engineering (`get_ichimoku_features`)
  - Training loop and backtest visualization
- `alpaca_drl_integration.py` — Live trading integration (currently a placeholder)
- `.env` — Must contain Alpaca API credentials

## Important conventions & assumptions 
- Feature shape: (sequence_length=30, input_size=6)
- Actions: 0=BUY, 1=SELL, 2=HOLD
- Training hyperparameters:
  - Replay buffer capacity: 10,000
  - Training batch size: 32 (increase if GPU memory allows, range: 32-128)
  - Learning starts when buffer ≥ 32 samples
  - Transaction cost: 0.001 (used in reward calculation)
- Model artifact: `drl_trading_model.pth` (contains model & optimizer state)

### Feature Normalization
- Cloud distance: Normalized relative to price
- Cross signals: Binary features (0, 1)
- Position indicators: Scaled to (-1, 0, 1)

### Common Issues
1. NaN features
   - Cause: Insufficient history for lookback periods
   - Fix: Ensure raw data length > max(52, 26)

2. Training instability
   - Cause: High learning rate or missing gradient clipping
   - Fix: Add `clip_grad_norm_(parameters, max_norm=1.0)`

3. Live trading gaps
   - Cause: Missed market data updates
   - Fix: Implement heartbeat in `on_minute_bar_update`

## Integration points & external deps
- `alpaca-py`: Crypto market data & trading API
  - `CryptoHistoricalDataClient` for training data
  - `TradingClient` for live orders (placeholder)
- Core Python packages:
  - `torch` for neural network & training
  - `pandas`, `numpy` for data processing
  - `matplotlib` for visualization
  - `python-dotenv` for API key management

## Developer workflows (concrete commands)

### Environment Setup & Validation
```python
def validate_environment():
    """Validate Python environment and dependencies."""
    import sys
    import torch
    from alpaca.data import CryptoHistoricalDataClient
    
    # Verify Python & CUDA
    print(f"Python: {sys.version}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Test Alpaca connection
    client = CryptoHistoricalDataClient(
        api_key=os.getenv('ALPACA_API_KEY_ID'),
        secret_key=os.getenv('ALPACA_API_SECRET_KEY')
    )
```

### Install & Train
```powershell
# Install dependencies
pip install torch numpy pandas alpaca-py python-dotenv matplotlib nest-asyncio

# Train and backtest (fetches 30 days by default)
python train_backtest.py
```

### Load Model for Inference
```python
from drl_agent import DRLTradingAgent
agent = DRLTradingAgent()
agent.load_model('drl_trading_model.pth')
action, name, prob = agent.select_action(state_sequence, deterministic=True)
```

### Basic Tests
```python
# Test feature calculation
def test_features():
    df = pd.DataFrame({
        'open': [100] * 60,
        'high': [110] * 60,
        'low': [90] * 60,
        'close': list(range(100, 160)),
        'volume': [1000] * 60
    })
    features = get_ichimoku_features(df)
    assert features['feature_price_cloud_position'].iloc[-1] == 1

# Test model determinism
def test_model():
    state = torch.randn(1, 30, 6)
    action1, _, _ = agent.select_action(state, deterministic=True) 
    action2, _, _ = agent.select_action(state, deterministic=True)
    assert action1 == action2
```

## Common development tasks
1. Implementing live trading integration:
   - Mirror `train_backtest.py` data prep in `alpaca_drl_integration.py`
   - Reuse `get_ichimoku_features` logic for feature parity
   - Load model with `deterministic=True` for inference
   - Add heartbeat monitoring for data gaps

2. Adding new features:
   - Extend `get_ichimoku_features` in both training and live code
   - Update `input_size` in `DRLTradingAgent`/`LSTMActorCritic`
   - Verify normalization for new features
   - Retrain model and validate performance

3. Modifying the model:
   - Core architecture in `LSTMActorCritic` class
   - Training loop in `train_agent` function
   - Add gradient clipping if needed
   - Ensure save/load compatibility

4. Performance optimization:
   - Tune batch size (32-128) based on GPU
   - Adjust sequence length (20-50) for pattern capture
   - Monitor replay buffer sampling efficiency
   - Profile feature calculation bottlenecks

## Feature implementation examples
- Ichimoku feature calculation: See `get_ichimoku_features` in `train_backtest.py`
- Action selection: `DRLTradingAgent.select_action` in `drl_agent.py` 
- Training loop: `train_agent` function in `train_backtest.py`
- Live trading placeholder: `alpaca_drl_integration.py` structure

## Future improvements
1. Model versioning:
   - Add version tracking in artifacts
   - Include training config in saves
   - Implement backward compatibility

2. Risk management:
   - Add position sizing logic
   - Implement stop-loss mechanisms
   - Add volatility-based filters

3. Online learning:
   - Enable continuous updates
   - Implement experience prioritization
   - Prevent catastrophic forgetting
