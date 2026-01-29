"""
Configuration file for DRL Trading System
All hyperparameters and settings in one place
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================
# API Configuration
# ============================================

API_CONFIG = {
    'api_key': os.getenv('ALPACA_API_KEY_ID'),
    'secret_key': os.getenv('ALPACA_API_SECRET_KEY'),
    'paper_trading': True,  # Set to False for live trading (be careful!)
}

# ============================================
# Market Configuration
# ============================================

MARKET_CONFIG = {
    'symbol': 'BTC/USD',
    'timeframe': 'Minute',  # Minute, Hour, Day
    'historical_days': 7,   # Days of historical data to fetch
}

# Alternative symbols to try:
# 'ETH/USD', 'SOL/USD', 'AVAX/USD', 'DOGE/USD', 'MATIC/USD'

# ============================================
# Feature Engineering Configuration
# ============================================

FEATURE_CONFIG = {
    # Ichimoku parameters
    'tenkan_period': 9,      # Conversion line period
    'kijun_period': 26,      # Base line period
    'senkou_b_period': 52,   # Leading span B period
    
    # Data preprocessing
    'outlier_multiplier': 1.5,  # IQR multiplier for outlier detection
    'fillna_method': 'ffill',   # Forward fill for missing values
    
    # Feature selection
    'features': [
        'feature_price_cloud_position',
        'feature_normalized_distance_from_cloud',
        'feature_tk_cross',
        'feature_kumo_twist',
        'feature_kumo_thickness_normalized',
        'feature_chikou_position'
    ],
}

# ============================================
# Neural Network Architecture
# ============================================

NETWORK_CONFIG = {
    'input_size': 6,              # Number of input features
    'hidden_size': 128,           # LSTM hidden dimension
    'num_lstm_layers': 2,         # Number of stacked LSTM layers
    'dropout': 0.2,               # Dropout rate for regularization
    'actor_hidden': 64,           # Actor network hidden layer size
    'critic_hidden': 64,          # Critic network hidden layer size
}

# For more complex patterns, try:
# hidden_size=256, num_lstm_layers=3

# For faster training on limited hardware:
# hidden_size=64, num_lstm_layers=1

# ============================================
# Reinforcement Learning Configuration
# ============================================

RL_CONFIG = {
    # Sequence and memory
    'sequence_length': 30,        # Number of time steps for LSTM input
    'replay_buffer_size': 10000,  # Maximum transitions to store
    
    # Learning parameters
    'learning_rate': 0.0001,      # Adam optimizer learning rate
    'gamma': 0.99,                # Discount factor for future rewards
    'epsilon': 0.2,               # PPO clipping parameter
    
    # Training hyperparameters
    'batch_size': 32,             # Training batch size
    'epochs_per_update': 4,       # Number of epochs per PPO update
    'gradient_clip': 0.5,         # Gradient clipping threshold
    
    # Exploration vs exploitation
    'entropy_coefficient': 0.01,  # Bonus for action diversity
    'value_loss_coefficient': 0.5,  # Weight for critic loss
    
    # Action selection
    'deterministic_inference': False,  # Use deterministic actions in live trading
}

# Tuning tips:
# - Higher learning_rate (0.001): Faster learning but less stable
# - Lower learning_rate (0.00001): More stable but slower convergence
# - Higher gamma (0.995): More future-focused
# - Lower gamma (0.95): More immediate reward focused
# - Higher epsilon (0.3): More conservative policy updates
# - Lower epsilon (0.1): More aggressive policy updates

# ============================================
# Trading Environment Configuration
# ============================================

ENVIRONMENT_CONFIG = {
    # Portfolio settings
    'initial_balance': 10000.0,    # Starting capital
    'position_size': 1.0,          # Fraction of capital per trade (0-1)
    'transaction_cost': 0.001,     # 0.1% transaction cost
    
    # Risk management
    'max_drawdown': 0.5,           # Stop if portfolio drops 50%
    'stop_loss': None,             # Per-trade stop loss (None = disabled)
    'take_profit': None,           # Per-trade take profit (None = disabled)
    
    # Position rules
    'allow_short': True,           # Allow short positions
    'max_position': 1,             # Maximum number of concurrent positions
}

# Conservative settings:
# position_size=0.5, transaction_cost=0.002, max_drawdown=0.3

# Aggressive settings:
# position_size=1.0, transaction_cost=0.0005, max_drawdown=0.7

# ============================================
# Training Configuration
# ============================================

TRAINING_CONFIG = {
    'num_episodes': 100,           # Total training episodes
    'max_steps_per_episode': 1000, # Maximum steps per episode
    'training_data_days': 30,      # Days of historical data for training
    'validation_split': 0.2,       # Fraction of data for validation
    'early_stopping_patience': 10, # Episodes without improvement before stopping
    'save_frequency': 10,          # Save model every N episodes
    'print_frequency': 10,         # Print progress every N episodes
}

# Quick testing:
# num_episodes=20, max_steps_per_episode=200

# Deep training:
# num_episodes=500, max_steps_per_episode=2000

# ============================================
# Live Trading Configuration
# ============================================

LIVE_TRADING_CONFIG = {
    'enable_trading': True,        # Actually execute trades (vs. simulation)
    'train_online': True,          # Continue learning during live trading
    'online_train_frequency': 10,  # Train every N bars
    'save_frequency': 100,         # Save model every N bars
    'log_frequency': 1,            # Log every N bars
    
    # Safety features
    'max_trades_per_hour': 10,     # Rate limit on trades
    'min_time_between_trades': 60, # Minimum seconds between trades
    'require_confirmation': False,  # Manual confirmation for each trade
}

# Paper trading mode (recommended for testing):
# enable_trading=False, train_online=True

# Full automation:
# enable_trading=True, train_online=True, require_confirmation=False

# ============================================
# Model Persistence Configuration
# ============================================

MODEL_CONFIG = {
    'model_path': 'drl_trading_model.pth',
    'checkpoint_dir': 'checkpoints/',
    'load_pretrained': True,       # Load existing model if available
    'save_best_only': False,       # Save only when performance improves
    'backup_frequency': 50,        # Backup model every N saves
}

# ============================================
# Logging Configuration
# ============================================

LOGGING_CONFIG = {
    'log_level': 'INFO',           # DEBUG, INFO, WARNING, ERROR
    'log_to_file': True,
    'log_file': 'trading_system.log',
    'log_format': '%(asctime)s - %(levelname)s - %(message)s',
    
    # What to log
    'log_features': True,          # Log feature values
    'log_actions': True,           # Log trading actions
    'log_rewards': True,           # Log rewards
    'log_training': True,          # Log training metrics
    'log_performance': True,       # Log performance metrics
}

# ============================================
# Visualization Configuration
# ============================================

VISUALIZATION_CONFIG = {
    'plot_training': True,         # Plot training curves
    'plot_backtest': True,         # Plot backtest results
    'save_plots': True,            # Save plots to files
    'plot_dir': 'plots/',
    'plot_format': 'png',          # png, jpg, svg, pdf
    'plot_dpi': 300,               # Resolution for saved plots
}

# ============================================
# Performance Metrics Configuration
# ============================================

METRICS_CONFIG = {
    'track_metrics': [
        'total_return',
        'sharpe_ratio',
        'max_drawdown',
        'win_rate',
        'profit_factor',
        'num_trades',
        'avg_trade_duration',
    ],
    
    # Benchmarks
    'benchmark': 'buy_and_hold',   # Comparison strategy
    'risk_free_rate': 0.02,        # Annual risk-free rate for Sharpe
}

# ============================================
# Hardware Configuration
# ============================================

HARDWARE_CONFIG = {
    'device': 'auto',              # 'auto', 'cpu', 'cuda', 'mps'
    'num_workers': 4,              # CPU workers for data loading
    'pin_memory': True,            # Speed up GPU transfers
    'use_mixed_precision': False,  # FP16 training (advanced)
}

# ============================================
# Debug Configuration
# ============================================

DEBUG_CONFIG = {
    'debug_mode': False,           # Enable verbose debugging
    'profile_code': False,         # Profile performance bottlenecks
    'check_gradients': False,      # Check for gradient issues
    'validate_features': True,     # Validate feature values
    'dry_run': False,              # Test without actual trading
}

# ============================================
# Preset Configurations
# ============================================

PRESETS = {
    'conservative': {
        'ENVIRONMENT_CONFIG': {
            'position_size': 0.3,
            'transaction_cost': 0.002,
            'max_drawdown': 0.3,
        },
        'RL_CONFIG': {
            'learning_rate': 0.00005,
            'epsilon': 0.3,
        },
        'LIVE_TRADING_CONFIG': {
            'max_trades_per_hour': 5,
        }
    },
    
    'aggressive': {
        'ENVIRONMENT_CONFIG': {
            'position_size': 0.8,
            'transaction_cost': 0.0005,
            'max_drawdown': 0.6,
        },
        'RL_CONFIG': {
            'learning_rate': 0.0003,
            'epsilon': 0.15,
        },
        'LIVE_TRADING_CONFIG': {
            'max_trades_per_hour': 20,
        }
    },
    
    'testing': {
        'TRAINING_CONFIG': {
            'num_episodes': 10,
            'max_steps_per_episode': 100,
        },
        'LIVE_TRADING_CONFIG': {
            'enable_trading': False,
            'train_online': False,
        },
        'DEBUG_CONFIG': {
            'debug_mode': True,
            'dry_run': True,
        }
    }
}

# ============================================
# Configuration Validation
# ============================================

def validate_config():
    """Validate configuration parameters."""
    errors = []
    
    # Check API keys
    if not API_CONFIG['api_key'] or not API_CONFIG['secret_key']:
        errors.append("API keys not found in environment variables")
    
    # Check network config
    if NETWORK_CONFIG['input_size'] != len(FEATURE_CONFIG['features']):
        errors.append(f"Input size ({NETWORK_CONFIG['input_size']}) doesn't match number of features ({len(FEATURE_CONFIG['features'])})")
    
    # Check environment config
    if not 0 < ENVIRONMENT_CONFIG['position_size'] <= 1:
        errors.append("position_size must be between 0 and 1")
    
    if ENVIRONMENT_CONFIG['initial_balance'] <= 0:
        errors.append("initial_balance must be positive")
    
    # Check RL config
    if not 0 < RL_CONFIG['gamma'] <= 1:
        errors.append("gamma must be between 0 and 1")
    
    if RL_CONFIG['learning_rate'] <= 0:
        errors.append("learning_rate must be positive")
    
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ Configuration validated successfully")
        return True

def apply_preset(preset_name):
    """Apply a preset configuration."""
    if preset_name not in PRESETS:
        print(f"❌ Preset '{preset_name}' not found")
        return False
    
    preset = PRESETS[preset_name]
    print(f"✅ Applying preset: {preset_name}")
    
    # Update configurations
    for config_name, updates in preset.items():
        globals()[config_name].update(updates)
    
    return True

def print_config_summary():
    """Print a summary of current configuration."""
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"\n📊 Market: {MARKET_CONFIG['symbol']}")
    print(f"💰 Initial Capital: ${ENVIRONMENT_CONFIG['initial_balance']:,.2f}")
    print(f"🧠 Network: {NETWORK_CONFIG['num_lstm_layers']} LSTM layers, {NETWORK_CONFIG['hidden_size']} hidden")
    print(f"📈 Sequence Length: {RL_CONFIG['sequence_length']} bars")
    print(f"🎓 Training Episodes: {TRAINING_CONFIG['num_episodes']}")
    print(f"⚡ Learning Rate: {RL_CONFIG['learning_rate']}")
    print(f"🎯 Position Size: {ENVIRONMENT_CONFIG['position_size']*100}%")
    print(f"🛡️  Max Drawdown: {ENVIRONMENT_CONFIG['max_drawdown']*100}%")
    print(f"💻 Device: {HARDWARE_CONFIG['device']}")
    print("="*60 + "\n")

# ============================================
# Usage Example
# ============================================

if __name__ == "__main__":
    # Validate configuration
    validate_config()
    
    # Print summary
    print_config_summary()
    
    # Example: Apply a preset
    # apply_preset('conservative')
    # print_config_summary()