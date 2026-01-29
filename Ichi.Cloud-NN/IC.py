import pandas as pd
import numpy as np

def get_ichimoku_features(df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52):
    """
    Calculates the six specified Ichimoku Cloud features for a given OHLC DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'High', 'Low', and 'Close' columns.
        tenkan_period (int): The look-back period for Tenkan-sen. Default is 9.
        kijun_period (int): The look-back period for Kijun-sen. Default is 26.
        senkou_b_period (int): The look-back period for Senkou Span B. Default is 52.

    Returns:
        pd.DataFrame: The original DataFrame augmented with the six Ichimoku features.
    """
    
    # --- 1. Calculate Core Ichimoku Components ---
    
    # Tenkan-sen (Conversion Line)
    tenkan_high = df['High'].rolling(window=tenkan_period).max()
    tenkan_low = df['Low'].rolling(window=tenkan_period).min()
    df['tenkan_sen'] = (tenkan_high + tenkan_low) / 2

    # Kijun-sen (Base Line)
    kijun_high = df['High'].rolling(window=kijun_period).max()
    kijun_low = df['Low'].rolling(window=kijun_period).min()
    df['kijun_sen'] = (kijun_high + kijun_low) / 2

    # Senkou Span A (Leading Span A)
    # This is plotted 26 periods in the future. We calculate it now and shift it later.
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(kijun_period)

    # Senkou Span B (Leading Span B)
    senkou_b_high = df['High'].rolling(window=senkou_b_period).max()
    senkou_b_low = df['Low'].rolling(window=senkou_b_period).min()
    df['senkou_span_b'] = ((senkou_b_high + senkou_b_low) / 2).shift(kijun_period)

    # Chikou Span (Lagging Span)
    # This is the close price plotted 26 periods in the past.
    df['chikou_span'] = df['Close'].shift(-kijun_period)
    
    # --- 2. Engineer the Six Features for the Neural Network ---

    # Feature 1: Price position relative to Kumo Cloud
    # Conditions: +1 if Close is above both Spans, -1 if below, 0 if inside.
    df['feature_price_cloud_position'] = np.where(
        df['Close'] > df['senkou_span_a'].combine_first(df['senkou_span_b']), 1, 
        np.where(df['Close'] < df['senkou_span_a'].combine_first(df['senkou_span_b']), -1, 0)
    )
    # Refine the condition for when spans cross
    df['feature_price_cloud_position'] = np.where(
        (df['Close'] > df['senkou_span_a']) & (df['Close'] > df['senkou_span_b']), 1,
        np.where((df['Close'] < df['senkou_span_a']) & (df['Close'] < df['senkou_span_b']), -1, 0)
    )


    # Feature 2: Normalized distance of price from the top/bottom of the cloud
    cloud_top = df[['senkou_span_a', 'senkou_span_b']].max(axis=1)
    cloud_bottom = df[['senkou_span_a', 'senkou_span_b']].min(axis=1)
    # Calculate distance to nearest cloud boundary
    distance_to_top = df['Close'] - cloud_top
    distance_to_bottom = df['Close'] - cloud_bottom
    # If price is above, distance is distance_to_top. If below, distance_to_bottom. Inside is 0.
    distance = np.where(df['feature_price_cloud_position'] == 1, distance_to_top, 
                        np.where(df['feature_price_cloud_position'] == -1, distance_to_bottom, 0))
    # Normalize by price to make it scale-invariant
    df['feature_normalized_distance_from_cloud'] = distance / df['Close']

    # Feature 3: Tenkan-sen / Kijun-sen cross
    # +1 for bullish cross, -1 for bearish cross, 0 otherwise
    tenkan_above = df['tenkan_sen'] > df['kijun_sen']
    # A cross occurs when the state (tenkan above/below) changes from the previous period
    cross_signal = (tenkan_above != tenkan_above.shift(1))
    df['feature_tk_cross'] = np.where(cross_signal & tenkan_above, 1, 
                                     np.where(cross_signal & ~tenkan_above, -1, 0))

    # Feature 4: Future Kumo twist status
    # +1 if the future cloud is bullish (Span A > Span B), -1 if bearish
    df['feature_kumo_twist'] = np.sign(df['senkou_span_a'] - df['senkou_span_b']).fillna(0)

    # Feature 5: Thickness of the future cloud (normalized)
    # A proxy for future volatility. Normalize by current close price.
    kumo_thickness = abs(df['senkou_span_a'] - df['senkou_span_b'])
    df['feature_kumo_thickness_normalized'] = kumo_thickness / df['Close']

    # Feature 6: Position of Chikou Span relative to price
    # +1 if Chikou is above the close price of its time, -1 if below.
    # Note: The comparison is between chikou_span and the Close price from 26 periods ago.
    price_for_chikou = df['Close'].shift(kijun_period) # Price at the time Chikou is plotted
    df['feature_chikou_position'] = np.sign(df['chikou_span'] - price_for_chikou).fillna(0)

    # Clean up intermediate columns before returning
    df_features = df[[
        'feature_price_cloud_position',
        'feature_normalized_distance_from_cloud',
        'feature_tk_cross',
        'feature_kumo_twist',
        'feature_kumo_thickness_normalized',
        'feature_chikou_position'
    ]].copy()

    return df_features

# --- Example Usage ---
if __name__ == '__main__':
    # Create a sample DataFrame with dummy OHLC data
    # In a real scenario, you would load this from a CSV or an API
    data = {
        'High': np.random.uniform(105, 125, 150),
        'Low': np.random.uniform(85, 105, 150),
        'Close': np.random.uniform(95, 115, 150)
    }
    sample_df = pd.DataFrame(data)
    sample_df['Open'] = (sample_df['High'] + sample_df['Low']) / 2 # Not needed for Ichimoku but good practice
    
    # Get the Ichimoku features
    ichimoku_features_df = get_ichimoku_features(sample_df)
    
    print("Generated Ichimoku Features:")
    # Displaying the last 10 rows to show calculated values (early rows will have NaNs)
    print(ichimoku_features_df.tail(10))

    # You can check for NaN values, which will appear at the start of the dataframe
    # due to the rolling windows. These rows should be dropped before training a model.
    print(f"\nNumber of rows with NaN values: {ichimoku_features_df.isnull().any(axis=1).sum()}")
    
    # Example of getting clean data for a model
    clean_features = ichimoku_features_df.dropna()
    print(f"\nShape of the feature set after dropping NaNs: {clean_features.shape}")