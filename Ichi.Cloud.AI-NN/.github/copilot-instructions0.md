## Purpose

Short, actionable notes for AI coding agents working on this DRL trading repo. Focus on the minimal context needed to be immediately productive: architecture, key files, conventions, workflows, and concrete examples.

## Big picture (what this repo does)
- Implements an LSTM-based Actor-Critic DRL trading agent trained with a simplified PPO flow
- Features are 6 Ichimoku-derived indicators (sequence length 30) used to predict Buy/Sell/Hold
- Training and backtesting use historical BTC/USD data via Alpaca's crypto API
- Model saves to `drl_trading_model.pth` for live inference/trading

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
  - Training batch size: 32
  - Learning starts when buffer ≥ 32 samples
  - Transaction cost: 0.001 (used in reward calculation)
- Model artifact: `drl_trading_model.pth` (contains model & optimizer state)

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

Install dependencies:
```powershell
pip install torch numpy pandas alpaca-py python-dotenv matplotlib nest-asyncio
```

Training and backtesting:
```powershell
python train_backtest.py  # Fetches 30 days by default
```

Load a saved model for inference:
```python
from drl_agent import DRLTradingAgent
agent = DRLTradingAgent()
agent.load_model('drl_trading_model.pth')
action, name, prob = agent.select_action(state_sequence, deterministic=True)
```

## Common development tasks
1. Implementing live trading integration:
   - Mirror `train_backtest.py` data prep in `alpaca_drl_integration.py`
   - Reuse `get_ichimoku_features` logic for feature parity
   - Load model with `deterministic=True` for inference

2. Adding new features:
   - Extend `get_ichimoku_features` in both training and live code
   - Update `input_size` in `DRLTradingAgent`/`LSTMActorCritic`
   - Retrain model to use new features

3. Modifying the model:
   - Core architecture in `LSTMActorCritic` class
   - Training loop in `train_agent` function
   - Ensure compatibility with existing save/load format

## Feature implementation examples
- Ichimoku feature calculation: See `get_ichimoku_features` in `train_backtest.py`
- Action selection: `DRLTradingAgent.select_action` in `drl_agent.py` 
- Training loop: `train_agent` function in `train_backtest.py`
- Live trading placeholder: `alpaca_drl_integration.py` structure
