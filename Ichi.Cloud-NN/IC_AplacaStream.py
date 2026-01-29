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

def get_ichimoku_features(df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52):
    """
    Calculates the six specified Ichimoku Cloud features.
    
    *** CORRECTED: Now uses lowercase 'high', 'low', 'close' to match Alpaca's DataFrame format. ***
    """
    df_temp = df.copy() # Work on a copy to avoid modifying the original ohlc data

    # --- 1. Calculate Core Ichimoku Components ---
    tenkan_high = df_temp['high'].rolling(window=tenkan_period).max()
    tenkan_low = df_temp['low'].rolling(window=tenkan_period).min()
    df_temp['tenkan_sen'] = (tenkan_high + tenkan_low) / 2

    kijun_high = df_temp['high'].rolling(window=kijun_period).max()
    kijun_low = df_temp['low'].rolling(window=kijun_period).min()
    df_temp['kijun_sen'] = (kijun_high + kijun_low) / 2
    
    # Senkou Span A is shifted into the future
    df_temp['senkou_span_a'] = ((df_temp['tenkan_sen'] + df_temp['kijun_sen']) / 2).shift(kijun_period)

    senkou_b_high = df_temp['high'].rolling(window=senkou_b_period).max()
    senkou_b_low = df_temp['low'].rolling(window=senkou_b_period).min()
    df_temp['senkou_span_b'] = ((senkou_b_high + senkou_b_low) / 2).shift(kijun_period)

    # Chikou Span is shifted into the past
    df_temp['chikou_span'] = df_temp['close'].shift(-kijun_period)

    # --- 2. Engineer the Six Features ---
    # Feature 1: Price position relative to Kumo Cloud
    cloud_top = df_temp[['senkou_span_a', 'senkou_span_b']].max(axis=1)
    cloud_bottom = df_temp[['senkou_span_a', 'senkou_span_b']].min(axis=1)
    df_temp['feature_price_cloud_position'] = np.where(df_temp['close'] > cloud_top, 1, 
                                                     np.where(df_temp['close'] < cloud_bottom, -1, 0))

    # Feature 2: Normalized distance of price from the cloud
    distance = np.where(df_temp['close'] > cloud_top, df_temp['close'] - cloud_top, 
                        np.where(df_temp['close'] < cloud_bottom, df_temp['close'] - cloud_bottom, 0))
    df_temp['feature_normalized_distance_from_cloud'] = distance / df_temp['close']

    # Feature 3: Tenkan-sen / Kijun-sen cross
    tenkan_above = df_temp['tenkan_sen'] > df_temp['kijun_sen']
    cross_signal = (tenkan_above != tenkan_above.shift(1))
    df_temp['feature_tk_cross'] = np.where(cross_signal & tenkan_above, 1, 
                                          np.where(cross_signal & ~tenkan_above, -1, 0))

    # Feature 4: Future Kumo twist status
    df_temp['feature_kumo_twist'] = np.sign(df_temp['senkou_span_a'] - df_temp['senkou_span_b']).fillna(0)

    # Feature 5: Normalized thickness of the future cloud
    kumo_thickness = abs(df_temp['senkou_span_a'] - df_temp['senkou_span_b'])
    df_temp['feature_kumo_thickness_normalized'] = kumo_thickness / df_temp['close']

    # Feature 6: Position of Chikou Span relative to price 26 periods ago
    price_for_chikou = df_temp['close'].shift(kijun_period)
    df_temp['feature_chikou_position'] = np.sign(df_temp['chikou_span'] - price_for_chikou).fillna(0)
    
    # Return only the specified feature columns
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
    """
    Master function to process the raw data and generate all features.
    This function should be called after any update to the raw data.
    """
    global data_store
    
    raw_df = data_store["raw_ohlc"]
    
    # 1. Pre-processing
    processed_df = raw_df.copy()
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    processed_df = handle_missing_values(processed_df, ohlcv_cols)
    processed_df = handle_outliers_iqr(processed_df, ohlcv_cols)
    
    # 2. Feature Engineering
    ichimoku_features = get_ichimoku_features(processed_df)
    
    # 3. Combine and Store
    # Here you would concatenate other feature sets (e.g., from other indicators)
    final_features = ichimoku_features # In the future: pd.concat([ichimoku_features, other_features], axis=1)
    
    # Drop rows with NaN values that result from rolling calculations
    final_features.dropna(inplace=True)
    
    data_store["features"] = final_features
    
    # --- For Demonstration: Print the latest feature set ---
    if not final_features.empty:
        print("\n--- Latest Feature Set ---")
        print(final_features.tail(1))
        # Here you would feed `final_features.tail(1)` into your trained Neural Network model.
    else:
        print("\n--- Feature set is empty. Waiting for more data to calculate features. ---")

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
        # Alpaca returns a multi-index, so we drop the 'symbol' level
        historical_data = historical_data.droplevel(0)
        
        # Rename columns for consistency
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
    asyncio.run(main())