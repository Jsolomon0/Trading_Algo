import os
import asyncio
import nest_asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import math
import matplotlib.pyplot as plt

# --- 1. Load Environment Variables ---
load_dotenv()
nest_asyncio.apply()

# --- 2. Configuration ---
API_KEY = os.getenv('ALPACA_API_KEY_ID')
SECRET_KEY = os.getenv('ALPACA_API_SECRET_KEY')
SYMBOL = "BTC/USD"
HISTORICAL_DAYS = 90 # Reduced for faster testing, 90 days is enough to avoid initial NaN issues

if not API_KEY or not SECRET_KEY:
    raise ValueError("Error: Alpaca API Key/Secret not found in environment variables.")

crypto_client = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)

# --- 3. Data Processing Functions (Unchanged) ---
def handle_missing_values(df, columns, method='ffill'):
    """Handles missing values in specified columns."""
    for col in columns:
        if df[col].isnull().any():
            df[col] = df[col].ffill()
    return df

def handle_outliers_iqr(df, columns, multiplier=1.5):
    """Handles outliers using the IQR method."""
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    return df

# --- 4. Fetch Historical Data Function (Unchanged) ---
async def fetch_historical_data(symbol, days):
    """Fetches historical minute bar data for the given symbol and number of days."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    request = CryptoBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Minute,
        start=start_date,
        end=end_date
    )
    bars = crypto_client.get_crypto_bars(request)
    if bars.df.empty:
        print(f"No historical data fetched for {symbol}.")
        return pd.DataFrame()
    historical_data = bars.df.droplevel(0)
    print(f"Fetched {len(historical_data)} historical bars for {symbol}.")
    return historical_data

# --- 5. CORRECTED Strategy Class ---
class IchimokuCloudStrategy:
    def __init__(self, data: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input 'data' must be a pandas DataFrame.")
        if not all(col in data.columns for col in ['high', 'low', 'close']):
            raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns.")
        # FIX: The check `max(senkou_b_period)` was incorrect. It should just check against the period value.
        if len(data) < senkou_b_period:
            raise ValueError("Insufficient data for the specified periods.")

        self.data = data.copy()
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period

        # This method will now be called to pre-calculate all features.
        self._calculate_features()

    def _calculate_features(self):
        """
        Calculates and attaches all Ichimoku features to the internal DataFrame.
        This is a private method called only during initialization.
        """
        # --- Calculate Core Ichimoku Components ---
        tenkan_high = self.data['high'].rolling(window=self.tenkan_period).max()
        tenkan_low = self.data['low'].rolling(window=self.tenkan_period).min()
        self.data['tenkan_sen'] = (tenkan_high + tenkan_low) / 2

        kijun_high = self.data['high'].rolling(window=self.kijun_period).max()
        kijun_low = self.data['low'].rolling(window=self.kijun_period).min()
        self.data['kijun_sen'] = (kijun_high + kijun_low) / 2

        self.data['senkou_span_a'] = ((self.data['tenkan_sen'] + self.data['kijun_sen']) / 2).shift(self.kijun_period)

        senkou_b_high = self.data['high'].rolling(window=self.senkou_b_period).max()
        senkou_b_low = self.data['low'].rolling(window=self.senkou_b_period).min()
        self.data['senkou_span_b'] = ((senkou_b_high + senkou_b_low) / 2).shift(self.kijun_period)

        self.data['chikou_span'] = self.data['close'].shift(-self.kijun_period)

        # --- Engineer the Six Features ---
        cloud_top = self.data[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        cloud_bottom = self.data[['senkou_span_a', 'senkou_span_b']].min(axis=1)

        self.data['feature_price_cloud_position'] = np.where(self.data['close'] > cloud_top, 1,
                                                             np.where(self.data['close'] < cloud_bottom, -1, 0))

        tenkan_above = self.data['tenkan_sen'] > self.data['kijun_sen']
        cross_signal = (tenkan_above != tenkan_above.shift(1))
        self.data['feature_tk_cross'] = np.where(cross_signal & tenkan_above, 1,
                                                 np.where(cross_signal & ~tenkan_above, -1, 0))

        price_for_chikou = self.data['close'].shift(self.kijun_period)
        self.data['feature_chikou_position'] = np.sign(self.data['chikou_span'] - price_for_chikou).fillna(0)

        # Drop rows with NaN values created by rolling windows to ensure signals are valid
        self.data.dropna(inplace=True)


    def generate_signal(self, i: int) -> str:
        """
        Generates a trading signal for a specific time step (row index i).
        This is the method the Backtester will call.

        LOGIC:
        - BUY Signal: A strong bullish confirmation. Price must be above the cloud,
          the Tenkan/Kijun cross must be bullish, AND the Chikou span must confirm the trend.
        - SELL Signal: A strong bearish confirmation. Price must be below the cloud,
          and the Tenkan/Kijun cross must be bearish.
        - HOLD: All other conditions.
        """
        # Get the row for the current timestep `i` using iloc
        row = self.data.iloc[i]

        # Bullish Conditions
        is_price_above_cloud = row['feature_price_cloud_position'] == 1
        is_tk_cross_bullish = row['feature_tk_cross'] == 1
        is_chikou_confirm_bullish = row['feature_chikou_position'] == 1

        # Bearish Conditions
        is_price_below_cloud = row['feature_price_cloud_position'] == -1
        is_tk_cross_bearish = row['feature_tk_cross'] == -1

        # Generate Signal
        if is_price_above_cloud and is_tk_cross_bullish and is_chikou_confirm_bullish:
            return "BUY"
        elif is_price_below_cloud and is_tk_cross_bearish:
            return "SELL"
        else:
            return "HOLD"


# --- 6. Backtester Class (with Plotting Fix) ---
class Backtester:
    # ... (The __init__ method is mostly the same, no changes needed there) ...
    def __init__(self, strategy_instance, initial_capital=100000, commission_rate=0.001, slippage_rate=0.0001, risk_free_rate=0.02):
        self.strategy = strategy_instance
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.risk_free_rate = risk_free_rate
        self.data = strategy_instance.data # Use the pre-processed data from the strategy
        if len(self.data) == 0:
            raise ValueError("Strategy data is empty. Cannot run backtest.")
        self.equity_curve = pd.Series(dtype=float)
        self.trades = []
        self.current_position_shares = 0
        self.entry_price = 0
        self.entry_time = None
        self.cash = initial_capital
        self.portfolio_value = initial_capital
        self.metrics = {}


    def run_backtest(self):
        """
        Runs the backtest simulation over the historical data.
        """
        if len(self.data) < 2:
            self.metrics = {"Error": "Not enough data points in strategy to run backtest."}
            return self.metrics

        self.equity_curve = pd.Series(index=self.data.index, dtype=float)
        self.equity_curve.iloc[0] = self.initial_capital

        for i in range(1, len(self.data)):
            current_timestamp = self.data.index[i]
            current_close = self.data['close'].iloc[i]

            # Update portfolio value based on the new price
            position_value = self.current_position_shares * current_close
            self.portfolio_value = self.cash + position_value
            self.equity_curve.iloc[i] = self.portfolio_value

            # Generate signal for the CURRENT index `i`
            signal = self.strategy.generate_signal(i)

            if signal == "BUY" and self.current_position_shares == 0:
                actual_entry_price = current_close * (1 + self.slippage_rate)
                shares_to_buy = self.cash / actual_entry_price
                commission = shares_to_buy * actual_entry_price * self.commission_rate
                
                self.current_position_shares = shares_to_buy
                self.entry_price = actual_entry_price
                self.entry_time = current_timestamp
                self.cash -= (shares_to_buy * actual_entry_price + commission)

            elif signal == "SELL" and self.current_position_shares > 0:
                actual_exit_price = current_close * (1 - self.slippage_rate)
                sale_value = self.current_position_shares * actual_exit_price
                commission = sale_value * self.commission_rate
                profit_loss = (actual_exit_price - self.entry_price) * self.current_position_shares - commission

                self.cash += sale_value - commission
                self.trades.append({
                    'type': 'BUY_SELL', 'entry_price': self.entry_price, 'exit_price': actual_exit_price,
                    'profit_loss': profit_loss, 'duration': (current_timestamp - self.entry_time).total_seconds() / 60,
                    'entry_time': self.entry_time, 'exit_time': current_timestamp, 'shares': self.current_position_shares
                })
                self.current_position_shares = 0
                self.entry_price = 0
                self.entry_time = None

        # Final cleanup
        if self.current_position_shares > 0:
            final_close_price = self.data['close'].iloc[-1]
            actual_exit_price = final_close_price * (1 - self.slippage_rate)
            sale_value = self.current_position_shares * actual_exit_price
            commission = sale_value * self.commission_rate
            profit_loss = (actual_exit_price - self.entry_price) * self.current_position_shares - commission
            
            self.cash += sale_value - commission
            self.trades.append({
                'type': 'FINAL_SELL', 'entry_price': self.entry_price, 'exit_price': actual_exit_price,
                'profit_loss': profit_loss, 'duration': (self.data.index[-1] - self.entry_time).total_seconds() / 60,
                'entry_time': self.entry_time, 'exit_time': self.data.index[-1], 'shares': self.current_position_shares
            })

        self.portfolio_value = self.cash
        self.equity_curve.iloc[-1] = self.portfolio_value
        
        self._calculate_metrics()
        self.plot_strategy() # Call plot after metrics are calculated
        return self.metrics

    def _calculate_metrics(self):
        # This function is complex and seems mostly correct, so we will leave it as is for now.
        # Minimal changes to ensure it runs without errors.
        metrics = {}
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            self.metrics = {"Error": "Equity curve is empty or has insufficient data."}
            return

        daily_equity = self.equity_curve.resample('D').last().ffill()
        daily_returns = daily_equity.pct_change().dropna()
        annualization_factor = 365

        total_return = (self.equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital
        metrics['Total Returns (%)'] = total_return * 100
        
        if not daily_returns.empty:
            annualized_return = (1 + daily_returns.mean())**annualization_factor - 1
            annualized_volatility = daily_returns.std() * np.sqrt(annualization_factor)
            
            excess_returns = daily_returns - (self.risk_free_rate / annualization_factor)
            metrics['Sharpe Ratio'] = (excess_returns.mean() / excess_returns.std()) * np.sqrt(annualization_factor) if excess_returns.std() != 0 else np.nan
        else:
            metrics['Sharpe Ratio'] = 0

        peak = self.equity_curve.expanding(min_periods=1).max()
        drawdown = (self.equity_curve - peak) / peak
        metrics['Max Drawdown (%)'] = drawdown.min() * 100 if not drawdown.empty else 0

        winning_trades = [t for t in self.trades if t['profit_loss'] > 0]
        total_trades = len(self.trades)
        metrics['Win Rate (%)'] = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0

        self.metrics = metrics
        return

    def plot_strategy(self):
        """
        FIXED: Plots the strategy performance and Ichimoku components.
        It no longer looks for 'Support' and 'Resistance'.
        """
        strategy_name = type(self.strategy).__name__
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
        
        # --- Plot 1: Price, Ichimoku Cloud, and Trades ---
        ax1.plot(self.data.index, self.data['close'], label='Close Price', color='black', alpha=0.9)
        ax1.fill_between(self.data.index, self.data['senkou_span_a'], self.data['senkou_span_b'],
                         where=self.data['senkou_span_a'] >= self.data['senkou_span_b'],
                         color='lightgreen', alpha=0.4, label='Bullish Kumo')
        ax1.fill_between(self.data.index, self.data['senkou_span_a'], self.data['senkou_span_b'],
                         where=self.data['senkou_span_a'] < self.data['senkou_span_b'],
                         color='lightcoral', alpha=0.4, label='Bearish Kumo')

        buy_signals = [trade['entry_time'] for trade in self.trades]
        sell_signals = [trade['exit_time'] for trade in self.trades]

        if buy_signals:
            buy_prices = self.data['close'].reindex(buy_signals, method='nearest')
            ax1.scatter(buy_prices.index, buy_prices, marker='^', s=100, color='green', label='Buy Signal', zorder=5)
        
        if sell_signals:
            sell_prices = self.data['close'].reindex(sell_signals, method='nearest')
            ax1.scatter(sell_prices.index, sell_prices, marker='v', s=100, color='red', label='Sell Signal', zorder=5)

        ax1.set_title(f'{strategy_name} Backtest: Trades and Ichimoku Cloud')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True)

        # --- Plot 2: Equity Curve ---
        ax2.plot(self.equity_curve.index, self.equity_curve, label='Equity Curve', color='blue')
        ax2.set_title('Portfolio Equity Curve')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Portfolio Value (USD)')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


# --- 7. Main Execution Logic (Corrected) ---
async def main():
    initial_data = await fetch_historical_data(SYMBOL, HISTORICAL_DAYS)

    if initial_data.empty:
        print("No historical data available. Exiting.")
        return

    # Preprocess data
    data_for_strategy = initial_data.copy()
    data_for_strategy = handle_missing_values(data_for_strategy, ['open', 'high', 'low', 'close', 'volume'])
    data_for_strategy = handle_outliers_iqr(data_for_strategy, ['open', 'high', 'low', 'close', 'volume'])

    # FIX: Initialize and run the correct strategy
    print("\n--- Running Backtest for Ichimoku Cloud Strategy ---")
    try:
        # Create the strategy instance. All feature calculation happens here.
        ichimoku_strategy = IchimokuCloudStrategy(data_for_strategy)

        # Pass the initialized strategy to the backtester
        backtester = Backtester(ichimoku_strategy, initial_capital=100000)
        
        # Run the backtest. The backtester will call the strategy's generate_signal method.
        metrics = backtester.run_backtest()

        if "Error" in metrics:
            print(f"Error during backtest: {metrics['Error']}")
            return

        print("\n--- Backtest Results ---")
        print(f"  Final Balance: ${backtester.portfolio_value:.2f}")
        print(f"  Total Returns: {metrics.get('Total Returns (%)', 0):.2f}%")
        print(f"  Win Rate: {metrics.get('Win Rate (%)', 0):.2f}%")
        print(f"  Max Drawdown: {metrics.get('Max Drawdown (%)', 0):.2f}%")
        print(f"  Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.2f}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())