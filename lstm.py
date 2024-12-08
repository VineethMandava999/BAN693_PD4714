import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
def set_random_seeds(seed_value=42):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

def get_stock_data(symbol, start_date, end_date):
    """Fetch stock data using yfinance."""
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

def prepare_data(data, n_steps):
    """Prepare the training data with a sliding window."""
    x, y = [], []
    for i in range(len(data) - n_steps):
        x.append(data[i:(i + n_steps), 0])
        y.append(data[i + n_steps, 0])
    return np.array(x), np.array(y)

def create_lstm_model(input_shape):
    """Create an LSTM-based model."""
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluate_regression_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, r2, mape

def predict_for_multiple_stocks_with_integers(stock_symbols, forecast_period=30, extended_forecast_period=180):
    """Predict stock prices for multiple stocks with integer results."""
    results = []
    overall_true = []
    overall_pred = []

    set_random_seeds()

    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')

    print("Starting LSTM-based stock prediction...")
    for symbol in stock_symbols:
        try:
            stock_data = get_stock_data(symbol, start_date, end_date)

            if stock_data.empty:
                print(f"No data for {symbol}. Skipping.")
                continue

            closing_prices = stock_data['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            closing_prices_scaled = scaler.fit_transform(closing_prices)

            if len(closing_prices_scaled) < 60:
                print(f"Not enough data for {symbol}. Skipping.")
                continue

            n_steps = 60
            x_train, y_train = prepare_data(closing_prices_scaled, n_steps)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            model = create_lstm_model((x_train.shape[1], 1))

            # Train the model with early stopping
            early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
            model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0, callbacks=[early_stopping])

            # Predict on training data for evaluation
            y_train_pred = model.predict(x_train)
            y_train_pred = scaler.inverse_transform(y_train_pred)
            y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))

            # Collect true and predicted values for overall metrics
            overall_true.extend(y_train_actual.flatten())
            overall_pred.extend(y_train_pred.flatten())

            # 1-month forecast (30 days)
            forecast_data = closing_prices_scaled[-n_steps:].copy()
            forecast = []
            for _ in range(forecast_period):
                forecast_input = np.reshape(forecast_data, (1, n_steps, 1))
                next_price_scaled = model.predict(forecast_input)[0][0]
                forecast.append(next_price_scaled)
                forecast_data = np.append(forecast_data[1:], [[next_price_scaled]], axis=0)

            forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

            # 6-month forecast (180 days)
            extended_forecast_data = closing_prices_scaled[-n_steps:].copy()
            extended_forecast = []
            for _ in range(extended_forecast_period):
                forecast_input = np.reshape(extended_forecast_data, (1, n_steps, 1))
                next_price_scaled = model.predict(forecast_input)[0][0]
                extended_forecast.append(next_price_scaled)
                extended_forecast_data = np.append(extended_forecast_data[1:], [[next_price_scaled]], axis=0)

            extended_forecast = scaler.inverse_transform(np.array(extended_forecast).reshape(-1, 1))

            # Extract numeric values for current price and forecasted profit/loss
            current_price = int(stock_data['Close'].iloc[-1])
            next_month_forecast = int(forecast[-1][0])
            next_six_months_forecast = int(extended_forecast[-1][0])
            profit_loss_next_month = int(next_month_forecast - current_price)
            profit_loss_next_six_months = int(next_six_months_forecast - current_price)

            results.append({
                'Symbol': symbol,
                'Current Price': current_price,
                'Forecasted Price (30 days)': next_month_forecast,
                'Profit/Loss (30 days)': profit_loss_next_month,
                'Forecasted Price (6 months)': next_six_months_forecast,
                'Profit/Loss (6 months)': profit_loss_next_six_months
            })

        except Exception as e:
            print(f"Error with {symbol}: {str(e)}")

    # Save results to CSV with integer values
    df_results = pd.DataFrame(results)
    df_results.to_csv('lstm_stock_predictions_with_integers1.csv', index=False)
    print("Predictions saved to 'lstm_stock_predictions_with_integers.csv'.")

    # Calculate and display overall regression metrics
    mae, rmse, r2, mape = evaluate_regression_metrics(np.array(overall_true), np.array(overall_pred))

    print(f"=== Overall Regression Metrics ===\n"
          f"MAE: {mae:.4f}\n"
          f"RMSE: {rmse:.4f}\n"
          f"R²: {r2:.4f}\n"
          f"MAPE: {mape:.4f}%")

# Example stock symbols
stock_symbols = [
    'AAPL', 'ABBV', 'ABT', 'ACN', 'ADP', 'AEE', 'AEP', 'ALL', 'AMGN', 'AMT', 'AMZN',
    'APTV', 'AON', 'AZO', 'BAC', 'BMY', 'BK', 'BKNG', 'BLK', 'BRK.B', 'C', 'CAT',
    'CB', 'CCI', 'CI', 'CL', 'CMCSA', 'CMS', 'COP', 'COST', 'CSCO', 'CTAS', 'DHR',
    'DHI', 'DIS', 'DLR', 'DLTR', 'DOW', 'DE', 'DUK', 'EOG', 'ED', 'ETR', 'EVRG',
    'EXC', 'EXPE', 'FIS', 'FISV', 'FCX', 'FE', 'GILD', 'GE', 'GLW', 'GOOG',
    'GOOGL', 'GS', 'HD', 'HCA', 'HLT', 'HON', 'IBM', 'INTC', 'ISRG', 'JNJ',
    'JPM', 'KMB', 'KR', 'KMI', 'LIN', 'LMT', 'LLY', 'MA', 'MCD', 'MCK', 'META',
    'MET', 'MMM', 'MPC', 'MRK', 'MS', 'MSFT', 'NKE', 'NOC', 'NVDA', 'ORCL', 'ORLY',
    'OXY', 'PAYX', 'PEP', 'PG', 'PLD', 'PNC', 'PNW', 'PPL', 'PFE', 'PSX', 'QCOM',
    'RTX', 'ROP', 'ROST', 'RSG', 'SBUX', 'SCHW', 'SO', 'SPG', 'SPGI', 'STZ', 'SYK',
    'SYY', 'T', 'TJX', 'TGT', 'TMO', 'TSLA', 'TXN', 'UNH', 'UNP', 'USB', 'V',
    'VLO', 'VTR', 'VZ', 'WBA', 'WEC', 'WELL', 'WMB', 'WMT', 'XEL', 'XOM'
]

# Predict and save the data
predict_for_multiple_stocks_with_integers(stock_symbols, forecast_period=30, extended_forecast_period=180)
