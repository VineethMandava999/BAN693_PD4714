import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Flatten

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

def create_transformer_model(input_shape, num_heads=4, ff_dim=128):
    """Create a Transformer-based model."""
    inputs = Input(shape=input_shape)

    # Multi-head attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[1])(inputs, inputs)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output + inputs)  # Residual connection

    # Feedforward network
    ffn = Dense(ff_dim, activation="relu")(attention_output)
    ffn = Dense(input_shape[1])(ffn)
    ffn_output = LayerNormalization(epsilon=1e-6)(ffn + attention_output)  # Residual connection

    # Final output layer
    flatten = Flatten()(ffn_output)
    outputs = Dense(1)(flatten)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def evaluate_model(y_true, y_pred):
    """Evaluate model performance using MAE, RMSE, R², and MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, r2, mape

def predict_for_multiple_stocks(stock_symbols, forecast_period=30):
    """Predict stock prices for multiple stocks and save results to CSV."""
    results = []

    set_random_seeds()

    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')

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

            model = create_transformer_model((x_train.shape[1], x_train.shape[2]))

            # Train the model
            model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

            # Predict on training data for evaluation
            y_train_pred = model.predict(x_train)
            y_train_pred = scaler.inverse_transform(y_train_pred)
            y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))

            # Performance metrics
            mae, rmse, r2, mape = evaluate_model(y_train_actual, y_train_pred)

            print(f"=== Overall Regression Metrics for {symbol} ===\n"
                  f"MAE: {mae:.4f}\n"
                  f"RMSE: {rmse:.4f}\n"
                  f"R²: {r2:.4f}\n"
                  f"MAPE: {mape:.4f}%")

            # 1-month forecast
            forecast_data = closing_prices_scaled[-n_steps:].copy()
            forecast = []

            for _ in range(forecast_period):
                forecast_input = np.reshape(forecast_data, (1, n_steps, 1))
                next_price_scaled = model.predict(forecast_input)[0][0]
                forecast.append(next_price_scaled)
                forecast_data = np.append(forecast_data[1:], [[next_price_scaled]], axis=0)

            forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
            next_month_forecast = forecast[-1][0]
            current_price = stock_data['Close'].iloc[-1]
            profit_loss_next_month = next_month_forecast - current_price

            results.append({
                'Symbol': symbol,
                'Current Price': int(current_price),
                'Forecasted Price (30 days)': int(next_month_forecast),
                'Profit/Loss (30 days)': int(profit_loss_next_month),
                'MAE': int(mae),
                'RMSE': int(rmse),
                'R²': int(r2 * 100),  # Represent as percentage
                'MAPE (%)': int(mape)
            })

            print(f"{symbol}: Prediction and evaluation completed.")

        except Exception as e:
            print(f"Error with {symbol}: {str(e)}")

    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv('transformer_stock_predictions_30_days.csv', index=False)
    print("All predictions saved to 'transformer_stock_predictions_30_days.csv'.")

# List of stock symbols (example set for faster testing)
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
predict_for_multiple_stocks(stock_symbols, forecast_period=30)
