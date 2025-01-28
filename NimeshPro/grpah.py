import sqlite3
import pandas as pd

# Path to SQLite database
db_path = "C:/Users/HP/Documents/NimeshPro/static/db.sqlite3"

# Attempt to connect to the database and check integrity
try:
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Perform a basic integrity check
    cursor.execute("PRAGMA integrity_check;")
    result = cursor.fetchone()

    if result[0] != 'ok':
        print(f"Database integrity check failed: {result[0]}")
        print("Database recovery is not implemented here. Please handle it manually.")
    
    # Query the data (e.g., closing prices)
    query = "SELECT * FROM stocks_stocks"  # Adjust table name as needed
    df = pd.read_sql_query(query, conn)

    # Display the DataFrame
    print(df)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Always close the connection
    if conn:
        conn.close()

# Query all tables in the SQLite database
conn = sqlite3.connect(db_path)
tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
tables = pd.read_sql_query(tables_query, conn)
conn.close()

# Display the tables
print(tables)

# Display the DataFrame
print(df)

# Count the occurrences of each symbol
symbol_counts = df['symbol'].value_counts()
print(symbol_counts)

# Load data
conn = sqlite3.connect(db_path)
financial_data_df = pd.read_sql("SELECT * FROM Stocks_financialdata", conn)
stocks_data_df = pd.read_sql("SELECT * FROM Stocks_stocks", conn)

# Merge the two tables on 'symbol'
merged_data = pd.merge(financial_data_df, stocks_data_df, on="symbol")
merged_data.head()

import plotly.graph_objects as go

# Plot graph for each symbol using closing price and EMA
import plotly.graph_objects as go

# Your existing code for generating the plot
symbols = merged_data['symbol'].unique()

fig = go.Figure()
for symbol in symbols:
    df_symbol = merged_data[merged_data['symbol'] == symbol]
    fig.add_trace(
        go.Scatter(
            x=df_symbol['date'],
            y=df_symbol['close_price'],
            name=f"{symbol} Close Price",
            visible=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_symbol['date'],
            y=df_symbol['ema20'],
            name=f"{symbol} EMA20",
            visible=False
        )
    )
default_button = dict(
    label="Search",  # The default label
    method="update",
    args=[{"visible": [False] * len(symbols) * 2}]  # Set all symbols to invisible
)

# Initialize an empty list for the buttons
buttons = [default_button]
buttons = []
for i, symbol in enumerate(symbols):
    visibility = [False] * len(symbols) * 2
    visibility[i * 2] = True
    visibility[i * 2 + 1] = True

    button = dict(
        label=symbol,
        method="update",
        args=[{"visible": visibility}]
    )
    buttons.append(button)

fig.update_layout(
    updatemenus=[dict(
        buttons=buttons,
        direction="down",
        showactive=True,
        x=0.1,
        y=1.1
    )]
)

# Export the graph to JSON
plot_json = fig.to_json()

import os
# Save the JSON as a file or pass it as needed
os.makedirs('templates', exist_ok=True)

# Write the JSON data to the 'assets/plot.json' file
with open('templates/plot.json', 'w') as f:
    f.write(plot_json)

# Calculate daily returns
merged_data_cleaned = merged_data.sort_values(by=['symbol', 'date'])
merged_data_cleaned['daily_return'] = merged_data_cleaned.groupby('symbol')['close_price'].pct_change() * 100
merged_data_cleaned = merged_data_cleaned.dropna(subset=['daily_return'])

# Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = merged_data_cleaned[['ema20', 'ema50', 'ema100', 'ema200', 'rsi']]
y = merged_data_cleaned['close_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM model (you must install tensorflow)
import tensorflow as tf
Model = tf.keras.models.Model
Sequential = tf.keras.models.Sequential
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
Input = tf.keras.layers.Input
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
LayerNormalization = tf.keras.layers.LayerNormalization
MultiHeadAttention = tf.keras.layers.MultiHeadAttention
Add = tf.keras.layers.Add
GlobalAveragePooling1D = tf.keras.layers.GlobalAveragePooling1D
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Prepare the dataset for LSTM
df_lstm = merged_data_cleaned[['date', 'close_price', 'ema20', 'ema50', 'ema100', 'ema200', 'rsi']]
df_lstm['date'] = pd.to_datetime(df_lstm['date'])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_lstm[['close_price', 'ema20', 'ema50', 'ema100', 'ema200', 'rsi']])

def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=1, batch_size=64, validation_data=(X_test, y_test))

# Predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)


import os
import http.server
import socketserver
import platform
import subprocess
import time

# Specify the port for the server
PORT = 8000

# Create an HTTP handler and server
Handler = http.server.SimpleHTTPRequestHandler

# Start the server in a separate process using subprocess
server_process = subprocess.Popen(["python", "-m", "http.server", str(PORT)])

# Allow a brief time for the server to start
time.sleep(1)

# Specify the path to your local HTML file
html_file_path = f'http://localhost:{PORT}/templates/dashboard.html'

# Check the operating system and open the file accordingly
if platform.system() == 'Windows':
    os.startfile(html_file_path)  # Windows
elif platform.system() == 'Darwin':  # macOS
    os.system(f'open {html_file_path}')
else:  # Linux and others
    os.system(f'xdg-open {html_file_path}')

# Wait for the server process to keep running
server_process.wait()

