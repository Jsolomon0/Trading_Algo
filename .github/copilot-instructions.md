# Copilot / AI Agent Instructions — Ichi.Cloud-NN project

This file is a compact, actionable guide to help AI coding agents be immediately productive in this repository.

1) Big-picture architecture
- Code lives mainly in the `Ichi.Cloud-NN/` folder. There are three logical concerns:
  - Feature engineering & strategy: `IC.py`, `IC_backtest.py` — implements Ichimoku feature extraction and a backtester.
  - DRL agent & environment: `drl_agent.py` — LSTM Actor-Critic network, ReplayBuffer, PPO-like training loop, and `TradingEnvironment` (reward logic).
  - Orchestration & I/O: `train_backtest.py` (training + backtest), `alpaca_drl_integration.py` (historical + live streaming via Alpaca).

2) Core data flow and conventions
- Input OHLCV DataFrame column names expected by most scripts: lowercase `open`, `high`, `low`, `close`, `volume` (some example helpers in `IC.py` also use capitalized names — prefer lowercase).
- Feature set: six Ichimoku-derived features saved / consumed under names:
  - `feature_price_cloud_position`, `feature_normalized_distance_from_cloud`, `feature_tk_cross`,
    `feature_kumo_twist`, `feature_kumo_thickness_normalized`, `feature_chikou_position`.
- Sequence length: default 30 time steps (see `SEQUENCE_LENGTH` in scripts and `RL_CONFIG.sequence_length` in `config.py`).
- Action mapping: 0 = BUY, 1 = SELL, 2 = HOLD (implemented in `drl_agent.DRLTradingAgent.action_map`).

3) Important files to reference
- `Ichi.Cloud-NN/config.py` — central place for hyperparameters, presets, and `validate_config()` / `print_config_summary()`.
- `Ichi.Cloud-NN/drl_agent.py` — network architecture (LSTMActorCritic), training (train_step), replay buffer, and `TradingEnvironment` reward logic.
- `Ichi.Cloud-NN/train_backtest.py` — example end-to-end training loop and backtest harness.
- `Ichi.Cloud-NN/IC.py` and `IC_backtest.py` — canonical Ichimoku feature implementation and backtester. Use these as the source of truth for feature names and rolling-window behavior.
- `Ichi.Cloud-NN/alpaca_drl_integration.py` — how live/historical Alpaca data is fetched and streamed; shows model loading/saving points (`drl_trading_model.pth`).

4) How to run (developer workflows)
- Important: change into the `Ichi.Cloud-NN` directory before running scripts so local imports resolve correctly.

  # Powershell example
  cd "Ichi.Cloud-NN"
  python .\train_backtest.py       # Train on historical data and produce 'drl_trading_model.pth'
  python .\IC_backtest.py          # Run the Ichimoku backtester (visualizes with matplotlib)
  python .\alpaca_drl_integration.py  # Start live streaming + optional online training (requires Alpaca keys)

- Environment variables: store `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY` in a `.env` at repository root (scripts call `load_dotenv()`).

5) Dependencies & environment notes
- Expected key packages (install in a virtualenv): python>=3.8, torch, pandas, numpy, python-dotenv, alpaca-py (Alpaca SDK), nest_asyncio, matplotlib. There is no requirements.txt in the repo — add one when modifying dependencies.
- Model file: `drl_trading_model.pth` is read/written by multiple scripts. Treat it as the canonical artifact.

6) Project-specific conventions and gotchas
- Many scripts assume the historical bars DataFrame returned by Alpaca has a top-level symbol index; code uses `.droplevel(0)` — preserve that when mocking data.
- Feature engineering creates NaNs due to rolling windows. Most pipelines call `.dropna()` before training/backtests; ensure enough history (>= 52 periods) before extracting features.
- Folder name `Ichi.Cloud-NN` contains dots and dashes; run scripts from inside that folder to avoid import path issues.
- `config.py` exposes `apply_preset('testing')` etc. Use presets to toggle safe testing modes (turn off live trading, reduce episodes, enable dry-run).

7) Integration points & extension hooks
- Alpaca: `CryptoHistoricalDataClient` for past bars and `CryptoDataStream` for live minute bars (see `alpaca_drl_integration.py`). Replace or mock these clients in unit tests.
- DRL agent API to call from orchestration:
  - initialize: instantiate `DRLTradingAgent(...)` and `TradingEnvironment(...)`
  - inference: `agent.select_action(state_sequence, deterministic=...)` returns `(action_idx, action_name, action_prob)`
  - persistence: `agent.save_model(path)` / `agent.load_model(path)`

8) Small examples for the agent (patterns to follow)
- Feature tensor shape for inference: (sequence_length, input_size) -> agent converts and unsqueezes to batch for the network.
- When creating features from historical CSV, ensure index is a timestamp and named `timestamp` (scripts expect a datetime index).

9) What not to change lightly
- The names of the six features and `sequence_length` are used across files — rename globally if you change them.
- Trading reward logic lives in `TradingEnvironment.calculate_reward` — changing it will alter all backtests and training results.

If anything is unclear or you'd like me to add quick unit tests, a requirements.txt, or to normalize the folder name to a valid Python package, tell me which area to focus on and I'll iterate.
