import os
import asyncio
import nest_asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.live import CryptoDataStream
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.models import Bar
from collections import deque

# Import the DRL agent (assuming it's in a file called drl_agent.py)
# from drl_agent import DRLTradingAgent, TradingEnvironment

# --- 1. Load Environment Variables ---
load_dotenv()
nest_asyncio.apply()

# --- 2. Configuration ---
API_KEY = os.getenv('ALPACA_API_KEY_ID')
SECRET_KEY = os.getenv('ALPACA_API_SECRET_KEY')
SYMBOL = "BTC/USD"
HISTORICAL_DAYS = 7
SEQUENCE_LENGTH = 30  # Number of timesteps for LSTM input

if not API_KEY or not SECRET_KEY:
    print("Error: API keys not found in environment variables.")
    exit(1)

# --- 3. Initialize Clients ---
crypto_stream = CryptoDataStream(api_key=API_KEY, secret_key=SECRET_KEY)
crypto_client = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)

# --- 4. Global Data Store ---
data_store = {
    "raw_ohlc": pd.DataFrame(),
    "features": pd.DataFrame(),
    "feature_sequence": deque(maxlen=SEQUENCE_LENGTH),  # Rolling window of features
    "agent": None,
    "environment": None,
    "trading_active": False,
    "last_action": "HOLD",
    "performance_metrics": {
        "total_rewards": 0,
        "num_trades": 0,
        "win_trades": 0,
        "loss_trades": 0
    }
}

# --- 5. Feature Engineering Functions ---

def handle_missing_values(df, columns, method='ffill'):
    """Handles missing values in specified columns."""
    for col in columns:
        if df[col].isnull().any():
            df[col] = df[col].ffill()
    return df

def handle_outliers_iqr(df, columns, multiplier=1.5):
    """Handles outliers using the IQR method by clipping values."""
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    return df

def get_ichimoku_features(df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52):
    """Calculates Ichimoku Cloud indicators and derives six features."""
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

def process_and_generate_features():
    """Master function to process raw data and generate features."""
    global data_store
    
    raw_df = data_store["raw_ohlc"]
    
    # 1. Pre-processing
    processed_df = raw_df.copy()
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    processed_df = handle_missing_values(processed_df, ohlcv_cols)
    processed_df = handle_outliers_iqr(processed_df, ohlcv_cols)
    
    # 2. Feature Engineering
    ichimoku_features = get_ichimoku_features(processed_df)
    
    # 3. Store features
    final_features = ichimoku_features
    final_features.dropna(inplace=True)
    
    data_store["features"] = final_features
    
    # 4. Update feature sequence for LSTM
    if not final_features.empty:
        latest_features = final_features.iloc[-1].values
        data_store["feature_sequence"].append(latest_features)
    
    return final_features

# --- 6. DRL Agent Integration ---

def initialize_drl_agent():
    """Initialize the Deep RL Trading Agent."""
    from drl_agent import DRLTradingAgent, TradingEnvironment
    
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
    
    # Try to load pre-trained model if it exists
    try:
        agent.load_model('drl_trading_model.pth')
        print("Loaded pre-trained model!")
    except:
        print("Starting with fresh model (no pre-trained weights found)")
    
    data_store["agent"] = agent
    data_store["environment"] = environment
    
    print("\n=== DRL Trading Agent Initialized ===")
    print(f"Device: {agent.device}")
    print(f"Network parameters: {sum(p.numel() for p in agent.network.parameters()):,}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print("=" * 40)

def make_trading_decision(current_price, prev_price):
    """
    Make a trading decision using the DRL agent.
    
    Args:
        current_price: Current BTC price
        prev_price: Previous BTC price
        
    Returns:
        action_name: Trading action (BUY, SELL, HOLD)
        action_prob: Probability of the action
        reward: Reward from the environment
    """
    global data_store
    
    agent = data_store["agent"]
    environment = data_store["environment"]
    feature_sequence = data_store["feature_sequence"]
    
    # Need at least SEQUENCE_LENGTH features for LSTM
    if len(feature_sequence) < SEQUENCE_LENGTH:
        return "HOLD", 0.0, 0.0
    
    # Convert deque to numpy array
    state_sequence = np.array(list(feature_sequence))
    
    # Get action from agent
    action, action_name, action_prob = agent.select_action(state_sequence, deterministic=False)
    
    # Calculate reward from environment
    reward, done = environment.calculate_reward(action, current_price, prev_price)
    
    # Store transition for training
    if len(feature_sequence) >= SEQUENCE_LENGTH:
        next_state = state_sequence  # Simplified - should be next state
        agent.store_transition(state_sequence, action, reward, next_state, done)
    
    # Update performance metrics
    data_store["performance_metrics"]["total_rewards"] += reward
    data_store["last_action"] = action_name
    
    if action_name != "HOLD":
        data_store["performance_metrics"]["num_trades"] += 1
        if reward > 0:
            data_store["performance_metrics"]["win_trades"] += 1
        else:
            data_store["performance_metrics"]["loss_trades"] += 1
    
    return action_name, action_prob, reward

def train_agent_periodically():
    """Train the agent periodically using accumulated experience."""
    agent = data_store["agent"]
    
    if len(agent.replay_buffer) >= 32:
        loss = agent.train_step(batch_size=32, epochs=4)
        if loss is not None:
            print(f"\n>>> Training Update: Loss = {loss:.4f}")
            
            # Save model periodically
            if len(agent.training_losses) % 10 == 0:
                agent.save_model('drl_trading_model.pth')

# --- 7. Data Fetching and Streaming ---

async def fetch_historical_data():
    """Fetches historical minute bar data."""
    global data_store
    print("--- Fetching Historical Data ---")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=HISTORICAL_DAYS)
    
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
        data_store["raw_ohlc"] = historical_data
        print(f"Successfully fetched {len(historical_data)} historical bars for {SYMBOL}.")
    else:
        print(f"No historical data found for {SYMBOL}.")

async def on_minute_bar_update(bar: Bar):
    """Callback for minute bar updates."""
    global data_store
    
    new_data = pd.DataFrame({
        'open': [bar.open],
        'high': [bar.high],
        'low': [bar.low],
        'close': [bar.close],
        'volume': [bar.volume]
    }, index=[bar.timestamp])
    
    # Append new bar to raw data
    data_store["raw_ohlc"] = pd.concat([data_store["raw_ohlc"], new_data])
    data_store["raw_ohlc"] = data_store["raw_ohlc"][~data_store["raw_ohlc"].index.duplicated(keep='last')]
    
    # Get previous price
    if len(data_store["raw_ohlc"]) >= 2:
        prev_price = data_store["raw_ohlc"].iloc[-2]['close']
    else:
        prev_price = bar.close
    
    current_price = bar.close
    
    # Generate features
    features = process_and_generate_features()
    
    if not features.empty:
        print(f"\n{'='*80}")
        print(f"Timestamp: {bar.timestamp}")
        print(f"BTC Price: ${current_price:,.2f}")
        print(f"{'='*80}")
        
        # Display latest features
        print("\n--- Latest Features ---")
        latest_features = features.tail(1)
        for col in latest_features.columns:
            print(f"{col:45s}: {latest_features[col].values[0]:+.6f}")
        
        # Make trading decision if agent is initialized and we have enough data
        if data_store["agent"] is not None and len(data_store["feature_sequence"]) >= SEQUENCE_LENGTH:
            action_name, action_prob, reward = make_trading_decision(current_price, prev_price)
            
            # Display trading decision
            print(f"\n--- Trading Decision ---")
            print(f"Action: {action_name} (Confidence: {action_prob*100:.1f}%)")
            print(f"Reward: {reward:+.6f}")
            
            # Display performance metrics
            metrics = data_store["performance_metrics"]
            win_rate = (metrics["win_trades"] / metrics["num_trades"] * 100) if metrics["num_trades"] > 0 else 0
            print(f"\n--- Performance Metrics ---")
            print(f"Total Rewards: {metrics['total_rewards']:+.4f}")
            print(f"Total Trades: {metrics['num_trades']}")
            print(f"Win Rate: {win_rate:.1f}% ({metrics['win_trades']}/{metrics['num_trades']})")
            print(f"Portfolio Value: ${data_store['environment'].portfolio_value:,.2f}")
            
            # Train agent every 10 bars
            bar_count = len(data_store["raw_ohlc"])
            if bar_count % 10 == 0:
                train_agent_periodically()
        else:
            print(f"\n⏳ Waiting for {SEQUENCE_LENGTH - len(data_store['feature_sequence'])} more bars to start trading...")
        
        print(f"{'='*80}\n")

# --- 8. Main Execution ---

async def main():
    """Main function to run the DRL trading system."""
    
    # Step 1: Fetch historical data
    await fetch_historical_data()
    
    # Step 2: Process historical data to build feature sequence
    print("\n--- Processing Historical Data ---")
    process_and_generate_features()
    print(f"Feature sequence length: {len(data_store['feature_sequence'])}/{SEQUENCE_LENGTH}")
    
    # Step 3: Initialize DRL agent
    initialize_drl_agent()
    
    # Step 4: Subscribe to live data
    print(f"\n--- Subscribing to Minute Bars for {SYMBOL} ---")
    crypto_stream.subscribe_bars(on_minute_bar_update, SYMBOL)
    
    # Step 5: Start streaming
    print("\n" + "="*80)
    print("🚀 LIVE DRL TRADING SYSTEM ACTIVE")
    print("="*80)
    print("Press Ctrl+C to stop the stream and save the model.\n")
    
    try:
        await crypto_stream.run()
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("⏹️  KeyboardInterrupt detected. Stopping system...")
        print("="*80)
    finally:
        # Save model before exiting
        if data_store["agent"] is not None:
            data_store["agent"].save_model('drl_trading_model.pth')
            
            # Print final statistics
            metrics = data_store["performance_metrics"]
            print("\n--- Final Statistics ---")
            print(f"Total Rewards: {metrics['total_rewards']:+.4f}")
            print(f"Total Trades: {metrics['num_trades']}")
            if metrics['num_trades'] > 0:
                win_rate = metrics["win_trades"] / metrics["num_trades"] * 100
                print(f"Win Rate: {win_rate:.1f}%")
            print(f"Final Portfolio Value: ${data_store['environment'].portfolio_value:,.2f}")
            print(f"Total Return: {(data_store['environment'].portfolio_value / data_store['environment'].initial_balance - 1) * 100:+.2f}%")
        
        await crypto_stream.close()
        print("\n✅ System stopped and closed gracefully.")

# --- 9. Run ---

if __name__ == "__main__":
    asyncio.run(main())
    