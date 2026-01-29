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

# --- 1. Load Environment Variables ---
load_dotenv()
nest_asyncio.apply()

# --- 2. Configuration ---
API_KEY = os.getenv('ALPACA_API_KEY_ID')
SECRET_KEY = os.getenv('ALPACA_API_SECRET_KEY')
SYMBOL = "BTC/USD"
HISTORICAL_DAYS = 7

# === [CHANGE 1: Define Lookback Periods] ===
KIJUN_PERIOD = 26 # The maximum lookback for most Ichimoku components and the shift period
SENKOU_B_PERIOD = 52 # The maximum lookback for all components
REQUIRED_BARS = SENKOU_B_PERIOD * 2 # Keep enough bars to cover the longest lookback plus some buffer

if not API_KEY or not SECRET_KEY:
    print("Error: ALPACA_API_KEY_ID or ALPACA_API_SECRET_KEY not found in environment variables.")
    exit(1)

# --- 3. Initialize Clients ---
crypto_stream = CryptoDataStream(api_key=API_KEY, secret_key=SECRET_KEY)
crypto_client = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)

# --- 4. Global DataFrame to Store Data ---
# Using a more robust approach with a global dictionary to hold our dataframes
data_store = {
    "raw_ohlc": pd.DataFrame(),
    "features": pd.DataFrame()
}

# --- 5. Data Processing & Feature Engineering Functions ---

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

def get_ichimoku_features(df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = KIJUN_PERIOD, senkou_b_period: int = SENKOU_B_PERIOD):
    """
    Calculates Ichimoku Cloud indicators and derives six features from them.
    
    LOGICAL ERROR 1 FIX: Corrected the Chikou Span logic.
    For feature engineering, we are interested in the signal at the current bar T.
    The signal is Close(T) vs Close(T-26).
    The Senkou Spans (A and B) are left with the forward shift, as the features (1, 2, 4, 5) 
    are meant to check price relative to the *future* cloud.
    """
    df_temp = df.copy()
    
    if len(df_temp) < senkou_b_period:
        # Not enough data for full Ichimoku calculation
        return pd.DataFrame()

    # ========================================
    # STEP 1: Calculate Ichimoku Components
    # ========================================
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_high_tenkan = df_temp['high'].rolling(window=tenkan_period).max()
    period_low_tenkan = df_temp['low'].rolling(window=tenkan_period).min()
    df_temp['tenkan_sen'] = (period_high_tenkan + period_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_high_kijun = df_temp['high'].rolling(window=kijun_period).max()
    period_low_kijun = df_temp['low'].rolling(window=kijun_period).min()
    df_temp['kijun_sen'] = (period_high_kijun + period_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods ahead
    # The shift is kept for the traditional definition, which informs the future cloud-based features (1, 2, 4, 5)
    df_temp['senkou_span_a'] = ((df_temp['tenkan_sen'] + df_temp['kijun_sen']) / 2).shift(kijun_period)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods ahead
    period_high_senkou_b = df_temp['high'].rolling(window=senkou_b_period).max()
    period_low_senkou_b = df_temp['low'].rolling(window=senkou_b_period).min()
    df_temp['senkou_span_b'] = ((period_high_senkou_b + period_low_senkou_b) / 2).shift(kijun_period)
    
    # ========================================
    # STEP 2: Calculate Features
    # ========================================
    
    # Feature 1: Price position relative to cloud
    df_temp['feature_price_cloud_position'] = np.where(
        (df_temp['close'] > df_temp['senkou_span_a']) & (df_temp['close'] > df_temp['senkou_span_b']), 
        1,
        np.where(
            (df_temp['close'] < df_temp['senkou_span_a']) & (df_temp['close'] < df_temp['senkou_span_b']), 
            -1, 
            0
        )
    )

    # Feature 2: Normalized distance of price from the top/bottom of the cloud
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

    # Feature 3: Tenkan-sen / Kijun-sen cross (sign of difference)
    # Note: Your original logic for feature_tk_cross was complex. This simpler version 
    # capturing the current state (Tenkan > Kijun) is often preferred for features.
    df_temp['feature_tk_cross'] = np.sign(df_temp['tenkan_sen'] - df_temp['kijun_sen']).fillna(0)
    # If you want the CROSS event (1 bar signal):
    # df_temp['feature_tk_cross'] = np.where(df_temp['tenkan_sen'].shift(1) < df_temp['kijun_sen'].shift(1), 1, 
    #                                  np.where(df_temp['tenkan_sen'].shift(1) > df_temp['kijun_sen'].shift(1), -1, 0))


    # Feature 4: Future Kumo twist status (sign of Senkou A vs B)
    df_temp['feature_kumo_twist'] = np.sign(df_temp['senkou_span_a'] - df_temp['senkou_span_b']).fillna(0)

    # Feature 5: Thickness of the future cloud (normalized)
    kumo_thickness = abs(df_temp['senkou_span_a'] - df_temp['senkou_span_b'])
    df_temp['feature_kumo_thickness_normalized'] = kumo_thickness / df_temp['close']

    # === [LOGICAL ERROR 1 FIX APPLIED HERE] ===
    # Feature 6: Position of Current Close price relative to the price 26 periods ago (Chikou signal)
    # A standard feature is np.sign(Current_Close - Close_26_Periods_Ago)
    price_26_periods_ago = df_temp['close'].shift(kijun_period)
    df_temp['feature_chikou_position'] = np.sign(df_temp['close'] - price_26_periods_ago).fillna(0)
    
    # Return only the specified feature columns
    feature_columns = [
        'feature_price_cloud_position',
        'feature_normalized_distance_from_cloud',
        'feature_tk_cross', # Reverted to sign for simplicity, but original cross logic was not strictly wrong.
        'feature_kumo_twist',
        'feature_kumo_thickness_normalized',
        'feature_chikou_position' # Corrected Logic
    ]
    return df_temp[feature_columns]

def process_and_generate_features():
    """
    Master function to process the raw data and generate all features.
    This function should be called after any update to the raw data.
    """
    global data_store
    
    raw_df = data_store["raw_ohlc"]
    
    # Check if there is enough data for the longest lookback period (52 bars)
    if len(raw_df) < SENKOU_B_PERIOD:
        print(f"\n--- Not enough data ({len(raw_df)} bars) to calculate Ichimoku features (requires at least {SENKOU_B_PERIOD}). ---")
        data_store["features"] = pd.DataFrame()
        return
        
    # 1. Pre-processing
    processed_df = raw_df.copy()
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    processed_df = handle_missing_values(processed_df, ohlcv_cols)
    # Outlier handling is often skipped or done on percentage change for live data
    # processed_df = handle_outliers_iqr(processed_df, ohlcv_cols)
    
    # 2. Feature Engineering
    ichimoku_features = get_ichimoku_features(processed_df)
    
    # 3. Combine and Store
    final_features = ichimoku_features
    
    # Only drop NaNs resulting from the lookback period to keep the feature set clean
    # The dropna is still here, but by limiting the size of 'raw_ohlc' in the streaming
    # function, we minimize the data frame's size and increase efficiency.
    final_features.dropna(inplace=True)
    
    data_store["features"] = final_features
    
    # --- For Demonstration: Print the latest feature set ---
    if not final_features.empty:
        print("\n--- Latest Feature Set ---")
        # Ensure we only print the last valid bar's feature vector
        print(final_features.tail(1))
        # Here you would feed `final_features.tail(1)` into your trained Neural Network model.
    else:
        print("\n--- Feature set is empty after processing. ---")


# --- 6. Data Fetching and Streaming Handlers ---

async def fetch_historical_data():
    """Fetches historical minute bar data to populate the initial DataFrame."""
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
        print(f"No historical data found for {SYMBOL} in the given range.")


async def on_minute_bar_update(bar: Bar):
    """Callback for minute bar data. Appends new data and triggers processing."""
    global data_store
    
    new_data = pd.DataFrame({
        'open': [bar.open],
        'high': [bar.high],
        'low': [bar.low],
        'close': [bar.close],
        'volume': [bar.volume]
    }, index=[bar.timestamp])
    
    # Append new bar to the raw data DataFrame
    data_store["raw_ohlc"] = pd.concat([data_store["raw_ohlc"], new_data])
    
    # Ensure there are no duplicate timestamps, keeping the last one
    data_store["raw_ohlc"] = data_store["raw_ohlc"][~data_store["raw_ohlc"].index.duplicated(keep='last')]
    
    # === [LOGICAL ERROR 2 FIX / Efficiency Improvement] ===
    # Limit the raw data size for efficiency. We only need bars covering the longest lookback (52)
    # plus the bar being processed. We use a buffer (e.g., 2 * 52) for safety.
    max_bars = REQUIRED_BARS
    if len(data_store["raw_ohlc"]) > max_bars:
        data_store["raw_ohlc"] = data_store["raw_ohlc"].iloc[-max_bars:]
    
    # Now, process the updated dataframe to get the latest features
    process_and_generate_features()


# --- 7. Main Execution Function ---

async def main():
    """Main function to set up subscriptions and run the stream."""
    await fetch_historical_data()
    
    print("\n--- Processing Initial Historical Data ---")
    process_and_generate_features() # Process the historical data once at the start

    print(f"\n--- Subscribing to Minute Bars for {SYMBOL} ---")
    crypto_stream.subscribe_bars(on_minute_bar_update, SYMBOL)
    
    print("\n--- Starting Crypto Data Stream ---")
    print("Press Ctrl+C to stop the stream gracefully.")

    try:
        await crypto_stream.run()
    except KeyboardInterrupt:
        print("\n--- KeyboardInterrupt detected. Stopping stream... ---")
    finally:
        await crypto_stream.close()
        print("--- Crypto Data Stream stopped and closed. ---")

# --- 8. Run the Main Function ---
if __name__ == "__main__":
    # The 'asyncio.run' will execute 'main'
    # NOTE: You must have your ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY 
    # correctly set in your .env file for this to work.
    asyncio.run(main())