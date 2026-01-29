import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from collections import deque
import torch

# Import DRL agent
from drl_agent import DRLTradingAgent, TradingEnvironment

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv('ALPACA_API_KEY_ID')
SECRET_KEY = os.getenv('ALPACA_API_SECRET_KEY')
SYMBOL = "BTC/USD"
SEQUENCE_LENGTH = 30

# ============================================
# Feature Engineering Functions
# ============================================

def handle_missing_values(df, columns):
    for col in columns:
        if df[col].isnull().any():
            df[col] = df[col].ffill()
    return df

def handle_outliers_iqr(df, columns, multiplier=1.5):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    return df

def get_ichimoku_features(df, tenkan_period=9, kijun_period=26, senkou_b_period=52):
    df_temp = df.copy()
    
    # Calculate Ichimoku Components
    period_high_tenkan = df_temp['high'].rolling(window=tenkan_period).max()
    period_low_tenkan = df_temp['low'].rolling(window=tenkan_period).min()
    df_temp['tenkan_sen'] = (period_high_tenkan + period_low_tenkan) / 2
    
    period_high_kijun = df_temp['high'].rolling(window=kijun_period).max()
    period_low_kijun = df_temp['low'].rolling(window=kijun_period).min()
    df_temp['kijun_sen'] = (period_high_kijun + period_low_kijun) / 2
    
    df_temp['senkou_span_a'] = ((df_temp['tenkan_sen'] + df_temp['kijun_sen']) / 2).shift(kijun_period)
    
    period_high_senkou_b = df_temp['high'].rolling(window=senkou_b_period).max()
    period_low_senkou_b = df_temp['low'].rolling(window=senkou_b_period).min()
    df_temp['senkou_span_b'] = ((period_high_senkou_b + period_low_senkou_b) / 2).shift(kijun_period)
    
    df_temp['chikou_span'] = df_temp['close'].shift(-kijun_period)
    
    # Calculate Features
    df_temp['feature_price_cloud_position'] = np.where(
        (df_temp['close'] > df_temp['senkou_span_a']) & (df_temp['close'] > df_temp['senkou_span_b']), 
        1,
        np.where(
            (df_temp['close'] < df_temp['senkou_span_a']) & (df_temp['close'] < df_temp['senkou_span_b']), 
            -1, 
            0
        )
    )
    
    cloud_top = df_temp[['senkou_span_a', 'senkou_span_b']].max(axis=1)
    cloud_bottom = df_temp[['senkou_span_a', 'senkou_span_b']].min(axis=1)
    distance_to_top = df_temp['close'] - cloud_top
    distance_to_bottom = df_temp['close'] - cloud_bottom
    distance = np.where(
        df_temp['feature_price_cloud_position'] == 1, 
        distance_to_top, 
        np.where(df_temp['feature_price_cloud_position'] == -1, distance_to_bottom, 0)
    )
    df_temp['feature_normalized_distance_from_cloud'] = distance / df_temp['close']
    
    tenkan_above = df_temp['tenkan_sen'] > df_temp['kijun_sen']
    cross_signal = (tenkan_above != tenkan_above.shift(1))
    df_temp['feature_tk_cross'] = np.where(
        cross_signal & tenkan_above, 
        1, 
        np.where(cross_signal & ~tenkan_above, -1, 0)
    )
    
    df_temp['feature_kumo_twist'] = np.sign(df_temp['senkou_span_a'] - df_temp['senkou_span_b']).fillna(0)
    
    kumo_thickness = abs(df_temp['senkou_span_a'] - df_temp['senkou_span_b'])
    df_temp['feature_kumo_thickness_normalized'] = kumo_thickness / df_temp['close']
    
    price_26_periods_ago = df_temp['close'].shift(kijun_period)
    df_temp['feature_chikou_position'] = np.sign(df_temp['chikou_span'] - price_26_periods_ago).fillna(0)
    
    feature_columns = [
        'feature_price_cloud_position',
        'feature_normalized_distance_from_cloud',
        'feature_tk_cross',
        'feature_kumo_twist',
        'feature_kumo_thickness_normalized',
        'feature_chikou_position'
    ]
    return df_temp[feature_columns]

# ============================================
# Data Preparation
# ============================================

def fetch_training_data(days=30):
    """Fetch historical data for training."""
    print(f"Fetching {days} days of historical data...")
    
    crypto_client = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    request = CryptoBarsRequest(
        symbol_or_symbols=[SYMBOL],
        timeframe=TimeFrame.Minute,
        start=start_date,
        end=end_date
    )
    
    bars = crypto_client.get_crypto_bars(request)
    historical_data = bars.df
    
    if not historical_data.empty:
        historical_data = historical_data.droplevel(0)
        historical_data.index.name = 'timestamp'
        print(f"✓ Fetched {len(historical_data)} bars")
        return historical_data
    else:
        print("✗ No data fetched!")
        return None

def prepare_features(raw_data):
    """Prepare features from raw OHLCV data."""
    print("Preparing features...")
    
    # Preprocessing
    df = raw_data.copy()
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    df = handle_missing_values(df, ohlcv_cols)
    df = handle_outliers_iqr(df, ohlcv_cols)
    
    # Feature engineering
    features = get_ichimoku_features(df)
    features.dropna(inplace=True)
    
    # Add price data for reward calculation
    features['close'] = df.loc[features.index, 'close']
    
    print(f"✓ Prepared {len(features)} feature vectors")
    return features

# ============================================
# Training Loop (FIXED)
# ============================================

def train_agent0(features_df, num_episodes=100, max_steps_per_episode=1000):
    """
    Train the DRL agent on historical data.
    """
    print("\n" + "="*60)
    print("TRAINING DRL AGENT")
    print("="*60)
    
    # Initialize agent and environment
    agent = DRLTradingAgent(
        input_size=6,
        hidden_size=128,
        num_lstm_layers=2,
        sequence_length=SEQUENCE_LENGTH,
        learning_rate=0.0001,
        gamma=0.99
    )
    
    environment = TradingEnvironment(
        initial_balance=10000.0,
        position_size=1.0,
        transaction_cost=0.001
    )
    
    # Prepare feature matrix
    feature_cols = [col for col in features_df.columns if col != 'close']
    feature_matrix = features_df[feature_cols].values
    prices = features_df['close'].values
    
    # Training metrics
    episode_rewards = []
    episode_portfolio_values = []
    episode_trades = []
    
    print(f"\nTraining configuration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Max steps per episode: {max_steps_per_episode}")
    print(f"  Training data points: {len(feature_matrix)}")
    print(f"  Device: {agent.device}")
    print("\nStarting training...\n")
    
    for episode in range(num_episodes):
        # Reset environment and agent
        environment.reset()
        agent.reset_hidden_state()
        
        # Randomly start somewhere in the data
        max_start = len(feature_matrix) - max_steps_per_episode - SEQUENCE_LENGTH
        if max_start <= 0:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, max_start)
        
        episode_reward = 0
        episode_trade_count = 0
        feature_sequence = deque(maxlen=SEQUENCE_LENGTH)
        
        # Initialize sequence
        for i in range(SEQUENCE_LENGTH):
            if start_idx + i < len(feature_matrix):
                feature_sequence.append(feature_matrix[start_idx + i])
        
        # Run episode
        for step in range(max_steps_per_episode):
            current_idx = start_idx + SEQUENCE_LENGTH + step
            
            if current_idx >= len(feature_matrix) - 1:
                break
            
            # Get current state
            state_sequence = np.array(list(feature_sequence))
            
            # Select action (STOCHASTIC for exploration during training)
            action, action_name, action_prob = agent.select_action(
                state_sequence, 
                deterministic=False  # IMPORTANT: False for training
            )
            
            # Get prices
            current_price = prices[current_idx]
            prev_price = prices[current_idx - 1]
            
            # Calculate reward
            reward, done = environment.calculate_reward(action, current_price, prev_price)
            episode_reward += reward
            
            # Track trades
            if action_name != "HOLD":
                episode_trade_count += 1
            
            # Get next state
            next_features = feature_matrix[current_idx]
            feature_sequence.append(next_features)
            next_state_sequence = np.array(list(feature_sequence))
            
            # Store transition
            agent.store_transition(state_sequence, action, reward, next_state_sequence, done)
            
            # Train agent periodically
            if len(agent.replay_buffer) >= 32 and step % 5 == 0:
                loss = agent.train_step(batch_size=32, epochs=4)
            
            if done:
                break
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_portfolio_values.append(environment.portfolio_value)
        episode_trades.append(episode_trade_count)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_portfolio = np.mean(episode_portfolio_values[-10:])
            avg_trades = np.mean(episode_trades[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:+.4f} | "
                  f"Avg Portfolio: ${avg_portfolio:,.2f} | "
                  f"Avg Trades/Ep: {avg_trades:.1f} | "
                  f"Buffer: {len(agent.replay_buffer)}")
    
    # Save trained model
    agent.save_model('drl_trading_model.pth')
    print("\n✓ Training complete! Model saved.")
    
    return agent, episode_rewards, episode_portfolio_values

# Temperature scaling for better action selection
def train_agent(features_df, num_episodes=100, max_steps_per_episode=1000):
    """
    Train the DRL agent on historical data.
    """
    print("\n" + "="*60)
    print("TRAINING DRL AGENT")
    print("="*60)
    
    # Initialize agent and environment
    agent = DRLTradingAgent(
        input_size=6,
        hidden_size=128,
        num_lstm_layers=2,
        sequence_length=SEQUENCE_LENGTH,
        learning_rate=0.0001,
        gamma=0.99
    )
    
    environment = TradingEnvironment(
        initial_balance=10000.0,
        position_size=1.0,
        transaction_cost=0.001
    )
    
    # Prepare feature matrix
    feature_cols = [col for col in features_df.columns if col != 'close']
    feature_matrix = features_df[feature_cols].values
    prices = features_df['close'].values
    
    # Training metrics
    episode_rewards = []
    episode_portfolio_values = []
    episode_trades = []
    
    print(f"\nTraining configuration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Max steps per episode: {max_steps_per_episode}")
    print(f"  Training data points: {len(feature_matrix)}")
    print(f"  Device: {agent.device}")
    print("\nStarting training...\n")
    
    for episode in range(num_episodes):
        # Reset environment and agent
        environment.reset()
        agent.reset_hidden_state()
        
        # Randomly start somewhere in the data
        max_start = len(feature_matrix) - max_steps_per_episode - SEQUENCE_LENGTH
        if max_start <= 0:
            start_idx = 0
        else:
            start_idx = np.random.randint(0, max_start)
        
        episode_reward = 0
        episode_trade_count = 0
        feature_sequence = deque(maxlen=SEQUENCE_LENGTH)
        
        # Initialize sequence
        for i in range(SEQUENCE_LENGTH):
            if start_idx + i < len(feature_matrix):
                feature_sequence.append(feature_matrix[start_idx + i])
        
        # Run episode
        for step in range(max_steps_per_episode):
            current_idx = start_idx + SEQUENCE_LENGTH + step
            
            if current_idx >= len(feature_matrix) - 1:
                break
            
            # Get current state
            state_sequence = np.array(list(feature_sequence))
            
            # Select action (STOCHASTIC for exploration during training)
            # More decisive backtest with temperature=0.5
            action, name, prob = agent.select_action(
                state_sequence, 
                deterministic=False, 
                temperature=0.5
                )
            
            # Get prices
            current_price = prices[current_idx]
            prev_price = prices[current_idx - 1]
            
            # Calculate reward
            reward, done = environment.calculate_reward(action, current_price, prev_price)
            episode_reward += reward
            
            # Track trades
            if action_name != "HOLD":
                episode_trade_count += 1
            
            # Get next state
            next_features = feature_matrix[current_idx]
            feature_sequence.append(next_features)
            next_state_sequence = np.array(list(feature_sequence))
            
            # Store transition
            agent.store_transition(state_sequence, action, reward, next_state_sequence, done)
            
            # Train agent periodically
            if len(agent.replay_buffer) >= 32 and step % 5 == 0:
                loss = agent.train_step(batch_size=32, epochs=4)
            
            if done:
                break
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_portfolio_values.append(environment.portfolio_value)
        episode_trades.append(episode_trade_count)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_portfolio = np.mean(episode_portfolio_values[-10:])
            avg_trades = np.mean(episode_trades[-10:])
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:+.4f} | "
                  f"Avg Portfolio: ${avg_portfolio:,.2f} | "
                  f"Avg Trades/Ep: {avg_trades:.1f} | "
                  f"Buffer: {len(agent.replay_buffer)}")
    
    # Save trained model
    agent.save_model('drl_trading_model.pth')
    print("\n✓ Training complete! Model saved.")
    
    return agent, episode_rewards, episode_portfolio_values

# ============================================
# Backtesting (FIXED)
# ============================================

def backtest_agent(agent, features_df, deterministic=True):
    """
    Backtest the trained agent on the data.
    
    FIXED: Now uses stochastic action selection if model is not well-trained
    """
    print("\n" + "="*60)
    print("BACKTESTING AGENT")
    print("="*60)
    
    environment = TradingEnvironment(initial_balance=10000.0)
    agent.reset_hidden_state()
    
    feature_cols = [col for col in features_df.columns if col != 'close']
    feature_matrix = features_df[feature_cols].values
    prices = features_df['close'].values
    
    feature_sequence = deque(maxlen=SEQUENCE_LENGTH)
    
    # Initialize sequence
    for i in range(SEQUENCE_LENGTH):
        feature_sequence.append(feature_matrix[i])
    
    # Tracking
    actions_taken = []
    action_details = []
    rewards = []
    portfolio_values = []
    positions = []
    
    # Run through data
    for i in range(SEQUENCE_LENGTH, len(feature_matrix)):
        state_sequence = np.array(list(feature_sequence))
        
        # IMPORTANT: Use deterministic=True for backtesting evaluation
        # But if you want to see more trades, set deterministic=False
        action, action_name, action_prob = agent.select_action(
            state_sequence, 
            deterministic=deterministic
        )
        
        # Calculate reward
        current_price = prices[i]
        prev_price = prices[i - 1]
        reward, done = environment.calculate_reward(action, current_price, prev_price)
        
        # Track metrics
        actions_taken.append(action_name)
        action_details.append({
            'timestamp': features_df.index[i],
            'action': action_name,
            'price': current_price,
            'confidence': action_prob,
            'position': environment.position
        })
        rewards.append(reward)
        portfolio_values.append(environment.portfolio_value)
        positions.append(environment.position)
        
        # Update sequence
        feature_sequence.append(feature_matrix[i])
    
    # Calculate statistics
    total_return = (environment.portfolio_value / environment.initial_balance - 1) * 100
    num_trades = len([a for a in actions_taken if a != 'HOLD'])
    num_buys = actions_taken.count('BUY')
    num_sells = actions_taken.count('SELL')
    num_holds = actions_taken.count('HOLD')
    
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"Initial Balance: ${environment.initial_balance:,.2f}")
    print(f"Final Balance: ${environment.portfolio_value:,.2f}")
    print(f"Total Return: {total_return:+.2f}%")
    print(f"\nActions Breakdown:")
    print(f"  Total Steps: {len(actions_taken)}")
    print(f"  BUY: {num_buys} ({num_buys/len(actions_taken)*100:.1f}%)")
    print(f"  SELL: {num_sells} ({num_sells/len(actions_taken)*100:.1f}%)")
    print(f"  HOLD: {num_holds} ({num_holds/len(actions_taken)*100:.1f}%)")
    print(f"  Total Trades (non-HOLD): {num_trades}")
    print(f"\nCumulative Reward: {sum(rewards):+.4f}")
    
    # Buy & Hold comparison
    buy_hold_return = (prices[-1] / prices[SEQUENCE_LENGTH] - 1) * 100
    print(f"\nBuy & Hold Return: {buy_hold_return:+.2f}%")
    print(f"Outperformance: {total_return - buy_hold_return:+.2f}%")
    print(f"{'='*60}\n")
    
    # Print sample of trades
    print("\nSample of Actions Taken:")
    print("-" * 80)
    trade_count = 0
    for detail in action_details[:100]:  # First 100 steps
        if detail['action'] != 'HOLD':
            print(f"{detail['timestamp']} | {detail['action']:4s} | "
                  f"${detail['price']:,.2f} | Conf: {detail['confidence']:.2f} | "
                  f"Pos: {detail['position']:+d}")
            trade_count += 1
            if trade_count >= 10:  # Show first 10 trades
                break
    print("-" * 80)
    
    return {
        'actions': actions_taken,
        'action_details': action_details,
        'rewards': rewards,
        'portfolio_values': portfolio_values,
        'positions': positions,
        'timestamps': features_df.index[SEQUENCE_LENGTH:]
    }

# ============================================
# Visualization
# ============================================

def plot_results(training_rewards, training_portfolios, backtest_results):
    """Plot training and backtest results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training rewards
    axes[0, 0].plot(training_rewards)
    axes[0, 0].set_title('Training Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    # Training portfolio values
    axes[0, 1].plot(training_portfolios)
    axes[0, 1].axhline(y=10000, color='r', linestyle='--', label='Initial Balance')
    axes[0, 1].set_title('Training Episode Portfolio Values')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Portfolio Value ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Backtest portfolio value
    axes[1, 0].plot(backtest_results['timestamps'], backtest_results['portfolio_values'])
    axes[1, 0].axhline(y=10000, color='r', linestyle='--', label='Initial Balance')
    axes[1, 0].set_title('Backtest Portfolio Value Over Time')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Portfolio Value ($)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Action distribution
    actions = backtest_results['actions']
    action_counts = {
        'BUY': actions.count('BUY'),
        'SELL': actions.count('SELL'),
        'HOLD': actions.count('HOLD')
    }
    axes[1, 1].bar(action_counts.keys(), action_counts.values(), color=['green', 'red', 'gray'])
    axes[1, 1].set_title('Action Distribution in Backtest')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True, axis='y')
    
    # Add percentages on bars
    for action, count in action_counts.items():
        pct = count / len(actions) * 100
        axes[1, 1].text(action, count, f'{pct:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('drl_training_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Results plot saved as 'drl_training_results.png'")
    plt.show()

# ============================================
# Main Execution
# ============================================

def main():
    print("\n" + "="*60)
    print("DRL TRADING AGENT - TRAINING & BACKTESTING")
    print("="*60 + "\n")
    
    # Step 1: Fetch data
    raw_data = fetch_training_data(days=30)
    if raw_data is None:
        return
    
    # Step 2: Prepare features
    features_df = prepare_features(raw_data)
    
    # Step 3: Train agent
    agent, episode_rewards, episode_portfolios = train_agent(
        features_df,
        num_episodes=100,
        max_steps_per_episode=1000
    )
    
    # Step 4: Backtest (try both modes)
    print("\n" + "="*60)
    print("Running backtest with DETERMINISTIC actions...")
    print("="*60)
    backtest_results_det = backtest_agent(agent, features_df, deterministic=True)
    
    print("\n" + "="*60)
    print("Running backtest with STOCHASTIC actions...")
    print("="*60)
    backtest_results_stoch = backtest_agent(agent, features_df, deterministic=False)
    
    # Step 5: Visualize
    plot_results(episode_rewards, episode_portfolios, backtest_results_stoch)
    
    print("\n" + "="*60)
    print("✓ ALL DONE!")
    print("="*60)
    print("\nYou can now use 'drl_trading_model.pth' for live trading.")
    print("If backtest shows few trades, the model may need more training episodes.")

if __name__ == "__main__":
    main()