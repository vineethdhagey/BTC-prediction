from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import joblib

def run_prediction(hours_ahead=24):
    # Load model and scaler
    model = load_model('lstm_model.h5')
    scaler = joblib.load('scaler.save')

    # Get last 10 days of hourly BTC data
    end = datetime.now()
    start = end - timedelta(days=30)
    btc_data = yf.download('BTC-USD', start=start, end=end, interval='1h')
    btc_data.reset_index(inplace=True)
    btc_data['Datetime'] = pd.to_datetime(btc_data['Datetime']).dt.tz_localize(None)

    # Simulate hourly sentiment (or replace with real-time fetching if possible)
    sentiment_score = 0  # fallback to neutral
    sentiment_series = pd.Series([sentiment_score] * len(btc_data), index=btc_data.index)
    btc_data['sentiment'] = sentiment_series
    btc_data['hour'] = btc_data['Datetime'].dt.floor('H')

    data = btc_data[['Close', 'sentiment']]
    scaled = scaler.transform(data)

    time_steps = 90
    last_sequence = scaled[-time_steps:]
    forecast_input = last_sequence.copy()
    forecast_prices = []

    for _ in range(hours_ahead):
        input_seq = np.expand_dims(forecast_input[-time_steps:], axis=0)
        pred = model.predict(input_seq, verbose=0)[0][0]
        forecast_input = np.vstack([forecast_input, [pred, forecast_input[-1][1]]])
        forecast_prices.append(pred)

    current_scaled_price = scaled[-1][0]
    predicted_scaled_price = forecast_prices[-1]

    current_price = scaler.inverse_transform([[current_scaled_price, 0]])[0][0]
    predicted_price = scaler.inverse_transform([[predicted_scaled_price, 0]])[0][0]

    return current_price, predicted_price
