import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
def set_random_seeds(seed_value=42):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

def get_stock_data(symbol, start_date, end_date):
    """Fetch stock data using yfinance."""
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    stock_data['MA_10'] = stock_data['Close'].rolling(window=10).mean()
    stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['RSI'] = compute_rsi(stock_data['Close'])
    stock_data.dropna(inplace=True)
    return stock_data

def compute_rsi(series, period=14):
    """Compute Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_data(data, n_steps):
    """Prepare the training data with a sliding window."""
    x, y = [], []
    for i in range(len(data) - n_steps):
        x.append(data[i:(i + n_steps)])
        y.append(data[i + n_steps, 0])  # Predicting 'Close' price
    return np.array(x), np.array(y)

def create_transformer_model(input_shape, num_heads=8, ff_dim=256, dropout_rate=0.2):
    """Create a Transformer-based model."""
    inputs = Input(shape=input_shape)

    # Multi-head attention
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output + inputs)

    # Feedforward network
    ffn = Dense(ff_dim, activation="relu")(attention_output)
    ffn = Dropout(dropout_rate)(ffn)
    ffn = Dense(input_shape[-1])(ffn)
    ffn_output = LayerNormalization(epsilon=1e-6)(ffn + attention_output)

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

def predict_for_multiple_stocks(stock_symbols, n_steps=60, forecast_period=30):
    """Predict stock prices for multiple stocks and display overall metrics."""
    results = []
    overall_metrics = {'MAE': [], 'RMSE': [], 'R²': [], 'MAPE': []}

    set_random_seeds()

    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')

    for symbol in stock_symbols:
        try:
            stock_data = get_stock_data(symbol, start_date, end_date)
            features = stock_data[['Close', 'Volume', 'MA_10', 'MA_50', 'RSI']].values

            scaler = MinMaxScaler(feature_range=(0, 1))
            features_scaled = scaler.fit_transform(features)

            x_train, y_train = prepare_data(features_scaled, n_steps)

            model = create_transformer_model((x_train.shape[1], x_train.shape[2]))

            # Train the model with early stopping
            early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1, callbacks=[early_stopping])

            # Predict on training data for evaluation
            y_train_pred = model.predict(x_train).flatten()
            y_train_actual = y_train.flatten()

            y_train_pred_actual = scaler.inverse_transform(
                np.concatenate([y_train_pred.reshape(-1, 1), np.zeros((len(y_train_pred), features_scaled.shape[1] - 1))], axis=1)
            )[:, 0]

            y_train_actual_values = scaler.inverse_transform(
                np.concatenate([y_train_actual.reshape(-1, 1), np.zeros((len(y_train_actual), features_scaled.shape[1] - 1))], axis=1)
            )[:, 0]

            mae, rmse, r2, mape = evaluate_model(y_train_actual_values, y_train_pred_actual)

            # Store metrics for overall analysis
            overall_metrics['MAE'].append(mae)
            overall_metrics['RMSE'].append(rmse)
            overall_metrics['R²'].append(r2)
            overall_metrics['MAPE'].append(mape)

            print(f"=== Metrics for {symbol} ===\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, MAPE: {mape:.4f}%")

            # Forecast for the next 30 days
            forecast_data = features_scaled[-n_steps:].copy()
            forecast = []
            for _ in range(forecast_period):
                forecast_input = np.reshape(forecast_data, (1, n_steps, features_scaled.shape[1]))
                next_price_scaled = model.predict(forecast_input)[0][0]
                forecast.append(next_price_scaled)
                forecast_data = np.append(forecast_data[1:], [[next_price_scaled] + [0] * (features_scaled.shape[1] - 1)], axis=0)

            forecast_actual = scaler.inverse_transform(
                np.concatenate([np.array(forecast).reshape(-1, 1), np.zeros((forecast_period, features_scaled.shape[1] - 1))], axis=1)
            )[:, 0]

            current_price = stock_data['Close'].iloc[-1]
            next_month_forecast = forecast_actual[-1]
            profit_loss = next_month_forecast - current_price

            results.append({
                'Symbol': symbol,
                'Current Price': int(current_price),
                'Forecasted Price (30 days)': int(next_month_forecast),
                'Profit/Loss (30 days)': int(profit_loss)
            })

        except Exception as e:
            print(f"Error with {symbol}: {e}")

    # Calculate overall metrics
    print("\n=== Overall Metrics Across All Stocks ===")
    print(f"Average MAE: {np.mean(overall_metrics['MAE']):.4f}")
    print(f"Average RMSE: {np.mean(overall_metrics['RMSE']):.4f}")
    print(f"Average R²: {np.mean(overall_metrics['R²']):.4f}")
    print(f"Average MAPE: {np.mean(overall_metrics['MAPE']):.4f}%")

    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv('improved_stock_predictions_30_days.csv', index=False)
    print("\nPredictions saved to 'improved_stock_predictions_30_days.csv'.")

# Stock symbols list
stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Predict and save results
predict_for_multiple_stocks(stock_symbols)
