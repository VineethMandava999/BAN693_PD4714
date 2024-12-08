import pandas as pd

# Paths to the CSV files
lstm_file = '/Users/lokeshharinath/Downloads/stocks prediction/lstm_stock_predictions_with_metrics.csv'  # Replace with the path to your LSTM CSV file
transformer_file = '/Users/lokeshharinath/Downloads/stocks prediction/transformer_stocks_forecast_results_integers.csv'  # Replace with the path to your Transformer CSV file

# Load the CSV files
lstm_data = pd.read_csv(lstm_file)
transformer_data = pd.read_csv(transformer_file)

# Add a model column to each dataframe
lstm_data['Model'] = 'LSTM'
transformer_data['Model'] = 'Transformer'

# Combine the dataframes
merged_data = pd.concat([lstm_data, transformer_data], ignore_index=True)

# Save the merged dataframe to a new CSV file
output_file = 'merged_stock_predictions_with_models.csv'  # Replace with your desired output path
merged_data.to_csv(output_file, index=False)

print(f"Merged file saved to: {output_file}")
