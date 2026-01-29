import os
import asyncio
import nest_asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.live import CryptoDataStream
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.models import Bar
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from collections import deque

# Import the DRL agent
# from drl_agent import DRLTradingAgent, TradingEnvironment

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- 1. Load Environment Variables ---
load_dotenv()
nest_asyncio.apply()

# --- 2. Configuration ---
API_KEY = os.getenv('ALPACA_API_KEY_ID')
SECRET_KEY = os.getenv('ALPACA_API_SECRET_KEY')
SYMBOL = "BTC/USD"
HISTORICAL_DAYS = 7
SEQUENCE_LENGTH = 30

# Trading configuration
ENABLE_LIVE_TRADING = False  # Set to True to enable actual order execution
MIN_ORDER_SIZE = 0.001       # Minimum BTC order size
MAX_ORDER_SIZE = 0.1         # Maximum BTC order size
MIN_TIME_BETWEEN_TRADES = 300  # Minimum 5 minutes between trades

if not API_KEY or not SECRET_KEY:
    logger.error("API keys not found in environment variables.")
    exit(1)

# --- 3. Initialize Clients ---
crypto_stream = CryptoDataStream(api_key=API_KEY, secret_key=SECRET_KEY)
crypto_client = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)
trading_client = TradingClient(api_key=API_KEY, secret_key=SECRET_KEY, paper=True)  # paper=True for paper trading

# --- 4. Global Data Store ---
data_store = {
    "raw_ohlc": pd.DataFrame(),
    "features": pd.DataFrame(),
    "feature_sequence": deque(maxlen=SEQUENCE_LENGTH),
    "agent": None,
    "environment": None,
    "last_action": "HOLD",
    "last_trade_time": None,
    "current_position": 0,  # 0: No position, 1: Long, -1: Short
    "position_size": 0,
    "entry_price": 0,
    "performance_metrics": {
        "total_rewards": 0,
        "num_trades": 0,
        "win_trades": 0,
        "loss_trades": 0,
        "total_pnl": 0
    }
}

# ============================================
# Order Submission Function
# ============================================

def submit_order(
    trading_client: TradingClient,
    symbol: str,
    qty: float,
    side: str,
    order_type: str = "market",
    time_in_force: str = "gtc",
    limit_price: float = None,
    client_order_id: str = None
):
    """
    Submits an order to Alpaca.
    
    Args:
        trading_client: The Alpaca TradingClient instance
        symbol: The symbol to trade (e.g., "BTC/USD")
        qty: The quantity to trade
        side: "buy" or "sell"
        order_type: "market" or "limit"
        time_in_force: "gtc", "ioc", "day"
        limit_price: Required for limit orders
        client_order_id: Unique identifier for the order
        
    Returns:
        The order response object, or None if an error occurred
    """
    try:
        # Validate inputs
        if side not in ("buy", "sell"):
            raise ValueError("Invalid side. Must be 'buy' or 'sell'.")
        
        if order_type not in ("market", "limit"):
            raise ValueError("Invalid order_type. Must be 'market' or 'limit'.")
        
        if time_in_force not in ("gtc", "ioc", "day"):
            raise ValueError("Invalid time_in_force. Must be 'gtc', 'ioc', or 'day'.")
        
        if order_type == "limit" and limit_price is None:
            raise ValueError("limit_price is required for limit orders.")
        
        # Prepare the order request
        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
        
        if order_type == "market":
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce(time_in_force),
                client_order_id=client_order_id
            )
        else:  # limit order
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce(time_in_force),
                limit_price=limit_price,
                client_order_id=client_order_id
            )
        
        # Submit the order
        order_response = trading_client.submit_order(order_data=order_request)
        logger.info(f"{order_type.upper()} order submitted for {symbol}: {side.upper()} {qty} @ {limit_price if limit_price else 'market'}")
        logger.info(f"Order ID: {order_response.id}, Status: {order_response.status}")
        
        return order_response
        
    except Exception as e:
        logger.error(f"Error submitting order: {e}")
        return None


def get_account_info():
    """Get current account information."""
    try:
        account = trading_client.get_account()
        logger.info(f"Account Status: {account.status}")
        logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
        logger.info(f"Cash: ${float(account.cash):,.2f}")
        logger.info(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
        return account
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        return None


def get_current_position(symbol: str):
    """Get current position for a symbol."""
    try:
        positions = trading_client.get_all_positions()
        for position in positions:
            if position.symbol == symbol:
                logger.info(f"Current Position in {symbol}:")
                logger.info(f"  Qty: {position.qty}")
                logger.info(f"  Avg Entry Price: ${float(position.avg_entry_price):,.2f}")
                logger.info(f"  Current Price: ${float(position.current_price):,.2f}")
                logger.info(f"  Market Value: ${float(position.market_value):,.2f}")
                logger.info(f"  Unrealized P&L: ${float(position.unrealized_pl):,.2f} ({float(position.unrealized_plpc)*100:.2f}%)")
                return position
        logger.info(f"No position in {symbol}")
        return None
    except Exception as e:
        logger.error(f"Error getting position: {e}")
        return None


def calculate_order_size(current_price: float, account_balance: float, position_size_pct: float = 0.1):
    """
    Calculate the order size based on account balance and risk parameters.
    
    Args:
        current_price: Current price of the asset
        account_balance: Available account balance
        position_size_pct: Percentage of account to use (default 10%)
        
    Returns:
        Order size in asset units
    """
    order_value = account_balance * position_size_pct
    order_size = order_value / current_price
    
    # Clip to min/max
    order_size = max(MIN_ORDER_SIZE, min(order_size, MAX_ORDER_SIZE))
    
    # Round to appropriate precision (e.g., 4 decimals for BTC)
    order_size = round(order_size, 4)
    
    return order_size


def can_trade():
    """Check if enough time has passed since last trade."""
    global data_store
    
    if data_store["last_trade_time"] is None:
        return True
    
    time_since_last_trade = (datetime.now() - data_store["last_trade_time"]).total_seconds()
    return time_since_last_trade >= MIN_TIME_BETWEEN_TRADES


def execute_trading_action(action_name: str, current_price: float):
    """
    Execute a trading action (BUY, SELL, HOLD).
    
    Args:
        action_name: Trading action from DRL agent
        current_price: Current market price
        
    Returns:
        True if order was executed, False otherwise
    """
    global data_store
    
    if not ENABLE_LIVE_TRADING:
        logger.info(f"[SIMULATION MODE] Would execute: {action_name} at ${current_price:,.2f}")
        return False
    
    if action_name == "HOLD":
        return False
    
    if not can_trade():
        logger.warning(f"Trade rate limit: Must wait {MIN_TIME_BETWEEN_TRADES}s between trades")
        return False
    
    try:
        # Get account info
        account = get_account_info()
        if not account:
            return False
        
        buying_power = float(account.buying_power)
        
        # Get current position
        current_position = get_current_position(SYMBOL)
        has_position = current_position is not None
        
        # Calculate order size
        order_size = calculate_order_size(current_price, buying_power, position_size_pct=0.1)
        
        # Generate unique order ID
        order_id = f"drl_{action_name.lower()}_{int(datetime.now().timestamp())}"
        
        # Execute based on action
        if action_name == "BUY":
            if has_position:
                logger.info(f"Already have a position in {SYMBOL}, skipping BUY")
                return False
            
            logger.info(f"🟢 EXECUTING BUY: {order_size} {SYMBOL} @ ${current_price:,.2f}")
            order = submit_order(
                trading_client=trading_client,
                symbol=SYMBOL,
                qty=order_size,
                side="buy",
                order_type="market",
                time_in_force="gtc",
                client_order_id=order_id
            )
            
            if order:
                data_store["current_position"] = 1
                data_store["position_size"] = order_size
                data_store["entry_price"] = current_price
                data_store["last_trade_time"] = datetime.now()
                data_store["performance_metrics"]["num_trades"] += 1
                return True
        
        elif action_name == "SELL":
            if not has_position:
                logger.info(f"No position in {SYMBOL}, skipping SELL")
                return False
            
            # Get actual position size
            position_qty = float(current_position.qty)
            
            logger.info(f"🔴 EXECUTING SELL: {position_qty} {SYMBOL} @ ${current_price:,.2f}")
            order = submit_order(
                trading_client=trading_client,
                symbol=SYMBOL,
                qty=position_qty,
                side="sell",
                order_type="market",
                time_in_force="gtc",
                client_order_id=order_id
            )
            
            if order:
                # Calculate P&L
                if data_store["entry_price"] > 0:
                    pnl = (current_price - data_store["entry_price"]) / data_store["entry_price"] * 100
                    data_store["performance_metrics"]["total_pnl"] += pnl
                    
                    if pnl > 0:
                        data_store["performance_metrics"]["win_trades"] += 1
                    else:
                        data_store["performance_metrics"]["loss_trades"] += 1
                    
                    logger.info(f"💰 Trade P&L: {pnl:+.2f}%")
                
                data_store["current_position"] = 0
                data_store["position_size"] = 0
                data_store["entry_price"] = 0
                data_store["last_trade_time"] = datetime.now()
                data_store["performance_metrics"]["num_trades"] += 1
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error executing trading action: {e}")
        return False


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


def process_and_generate_features():
    """Master function to process raw data and generate features."""
    global data_store
    
    raw_df = data_store["raw_ohlc"]
    
    processed_df = raw_df.copy()
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    processed_df = handle_missing_values(processed_df, ohlcv_cols)
    processed_df = handle_outliers_iqr(processed_df, ohlcv_cols)
    
    ichimoku_features = get_ichimoku_features(processed_df)
    
    final_features = ichimoku_features
    final_features.dropna(inplace=True)
    
    data_store["features"] = final_features
    
    if not final_features.empty:
        latest_features = final_features.iloc[-1].values
        data_store["feature_sequence"].append(latest_features)
    
    return final_features


# ============================================
# DRL Agent Integration
# ============================================

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
    
    try:
        agent.load_model('drl_trading_model.pth')
        logger.info("✅ Loaded pre-trained model!")
    except:
        logger.warning("⚠️  Starting with fresh model (no pre-trained weights found)")
    
    data_store["agent"] = agent
    data_store["environment"] = environment
    
    logger.info("\n" + "="*60)
    logger.info("DRL TRADING AGENT INITIALIZED")
    logger.info("="*60)
    logger.info(f"Device: {agent.device}")
    logger.info(f"Network parameters: {sum(p.numel() for p in agent.network.parameters()):,}")
    logger.info(f"Sequence length: {SEQUENCE_LENGTH}")
    logger.info(f"Live Trading: {'ENABLED' if ENABLE_LIVE_TRADING else 'SIMULATION MODE'}")
    logger.info("="*60 + "\n")


def make_trading_decision(current_price, prev_price):
    """Make a trading decision using the DRL agent."""
    global data_store
    
    agent = data_store["agent"]
    environment = data_store["environment"]
    feature_sequence = data_store["feature_sequence"]
    
    if len(feature_sequence) < SEQUENCE_LENGTH:
        return "HOLD", 0.0, 0.0
    
    state_sequence = np.array(list(feature_sequence))
    
    action, action_name, action_prob = agent.select_action(state_sequence, deterministic=False)
    
    reward, done = environment.calculate_reward(action, current_price, prev_price)
    
    if len(feature_sequence) >= SEQUENCE_LENGTH:
        next_state = state_sequence
        agent.store_transition(state_sequence, action, reward, next_state, done)
    
    data_store["performance_metrics"]["total_rewards"] += reward
    data_store["last_action"] = action_name
    
    return action_name, action_prob, reward


def train_agent_periodically():
    """Train the agent periodically using accumulated experience."""
    agent = data_store["agent"]
    
    if len(agent.replay_buffer) >= 32:
        loss = agent.train_step(batch_size=32, epochs=4)
        if loss is not None:
            logger.info(f">>> Training Update: Loss = {loss:.4f}")
            
            if len(agent.training_losses) % 10 == 0:
                agent.save_model('drl_trading_model.pth')


# ============================================
# Data Fetching and Streaming
# ============================================

async def fetch_historical_data():
    """Fetches historical minute bar data."""
    global data_store
    logger.info("--- Fetching Historical Data ---")
    
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
        logger.info(f"✅ Successfully fetched {len(historical_data)} historical bars for {SYMBOL}.")
    else:
        logger.warning(f"❌ No historical data found for {SYMBOL}.")


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
    
    data_store["raw_ohlc"] = pd.concat([data_store["raw_ohlc"], new_data])
    data_store["raw_ohlc"] = data_store["raw_ohlc"][~data_store["raw_ohlc"].index.duplicated(keep='last')]
    
    if len(data_store["raw_ohlc"]) >= 2:
        prev_price = data_store["raw_ohlc"].iloc[-2]['close']
    else:
        prev_price = bar.close
    
    current_price = bar.close
    
    features = process_and_generate_features()
    
    if not features.empty:
        logger.info(f"\n{'='*80}")
        logger.info(f"Timestamp: {bar.timestamp}")
        logger.info(f"BTC Price: ${current_price:,.2f}")
        logger.info(f"{'='*80}")
        
        logger.info("\n--- Latest Features ---")
        latest_features = features.tail(1)
        for col in latest_features.columns:
            logger.info(f"{col:45s}: {latest_features[col].values[0]:+.6f}")
        
        if data_store["agent"] is not None and len(data_store["feature_sequence"]) >= SEQUENCE_LENGTH:
            action_name, action_prob, reward = make_trading_decision(current_price, prev_price)
            
            logger.info(f"\n--- Trading Decision ---")
            logger.info(f"Action: {action_name} (Confidence: {action_prob*100:.1f}%)")
            logger.info(f"Reward: {reward:+.6f}")
            
            # Execute the trading action
            executed = execute_trading_action(action_name, current_price)
            if executed:
                logger.info(f"✅ Trade executed successfully")
            
            # Display performance metrics
            metrics = data_store["performance_metrics"]
            win_rate = (metrics["win_trades"] / metrics["num_trades"] * 100) if metrics["num_trades"] > 0 else 0
            logger.info(f"\n--- Performance Metrics ---")
            logger.info(f"Total Rewards: {metrics['total_rewards']:+.4f}")
            logger.info(f"Total Trades: {metrics['num_trades']}")
            logger.info(f"Win Rate: {win_rate:.1f}% ({metrics['win_trades']}/{metrics['num_trades']})")
            logger.info(f"Total P&L: {metrics['total_pnl']:+.2f}%")
            logger.info(f"Current Position: {data_store['current_position']} ({data_store['position_size']:.4f} BTC)")
            logger.info(f"Portfolio Value: ${data_store['environment'].portfolio_value:,.2f}")
            
            bar_count = len(data_store["raw_ohlc"])
            if bar_count % 10 == 0:
                train_agent_periodically()
        else:
            logger.info(f"\n⏳ Waiting for {SEQUENCE_LENGTH - len(data_store['feature_sequence'])} more bars to start trading...")
        
        logger.info(f"{'='*80}\n")


# ============================================
# Main Execution
# ============================================

async def main():
    """Main function to run the DRL trading system."""
    
    # Display account info
    logger.info("\n" + "="*80)
    logger.info("ACCOUNT INFORMATION")
    logger.info("="*80)
    get_account_info()
    get_current_position(SYMBOL)
    logger.info("="*80 + "\n")
    
    # Fetch historical data
    await fetch_historical_data()
    
    # Process historical data
    logger.info("\n--- Processing Historical Data ---")
    process_and_generate_features()
    logger.info(f"Feature sequence length: {len(data_store['feature_sequence'])}/{SEQUENCE_LENGTH}")
    
    # Initialize DRL agent
    initialize_drl_agent()
    
    # Subscribe to live data
    logger.info(f"\n--- Subscribing to Minute Bars for {SYMBOL} ---")
    crypto_stream.subscribe_bars(on_minute_bar_update, SYMBOL)
    
    # Start streaming
    logger.info("\n" + "="*80)
    logger.info("🚀 LIVE DRL TRADING SYSTEM ACTIVE")
    logger.info("="*80)
    logger.info(f"Mode: {'LIVE TRADING' if ENABLE_LIVE_TRADING else 'SIMULATION'}")
    logger.info("Press Ctrl+C to stop the stream and save the model.\n")
    
    try:
        await crypto_stream.run()
    except KeyboardInterrupt:
        logger.info("\n\n" + "="*80)
        logger.info("⏹️  KeyboardInterrupt detected. Stopping system...")
        logger.info("="*80)
    finally:
        if data_store["agent"] is not None:
            data_store["agent"].save_model('drl_trading_model.pth')
            
            metrics = data_store["performance_metrics"]
            logger.info("\n--- Final Statistics ---")
            logger.info(f"Total Rewards: {metrics['total_rewards']:+.4f}")
            logger.info(f"Total Trades: {metrics['num_trades']}")
            if metrics['num_trades'] > 0:
                win_rate = metrics["win_trades"] / metrics["num_trades"] * 100
                logger.info(f"Win Rate: {win_rate:.1f}%")
            logger.info(f"Total P&L: {metrics['total_pnl']:+.2f}%")
            logger.info(f"Final Portfolio Value: ${data_store['environment'].portfolio_value:,.2f}")
        
        await crypto_stream.close()
        logger.info("\n✅ System stopped and closed gracefully.")


if __name__ == "__main__":
    asyncio.run(main())