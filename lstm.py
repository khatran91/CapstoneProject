import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime

def load_city_data(city_name):
    """Load the weather data for a specific city."""
    # file_path = f"data/{city_name}.csv"  # Update path as needed
    # weather = pd.read_csv(file_path, skiprows=2)
    weather = pd.read_csv("Omaha.csv", skiprows=2)
    weather['time'] = pd.to_datetime(weather['time'])
    weather['year'] = weather['time'].dt.year
    weather['week'] = weather['time'].dt.isocalendar().week
    return weather

def process_weekly_data(weather_data):
    """Calculate weekly average temperature."""
    return weather_data.groupby(['year', 'week'])['temperature_2m (°F)'].mean().reset_index()

def create_weekly_dataset(data, look_back=48):
    """Create sequences of data for LSTM training."""
    X, y = [], []
    for i in range(len(data) - look_back - 168):  # 168 hours = 1 week
        X.append(data[i:i + look_back, 0])
        y.append(data[i + look_back: i + look_back + 168, 0])
    return np.array(X), np.array(y)

def train_lstm_model(X, y, look_back=48):
    """Train the LSTM model."""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(look_back, 1)))
    model.add(Dense(units=168))  # Predict 168 hours (1 week)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)
    return model

def predict_weather(weather_data, start_date):
    """Predict weekly weather using an LSTM model."""
    # Parse the user-provided date
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    target_year, target_week = start_date.isocalendar()[:2]

    # Filter historical data for the same week in previous years
    historical_week_data = weather_data[(weather_data['week'] == target_week) & 
                                        (weather_data['year'] < target_year)]
    if historical_week_data.empty:
        raise ValueError("No historical data available for the selected week.")

    # Scale the temperature data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_temperature = scaler.fit_transform(historical_week_data[['temperature_2m (°F)']])

    # Prepare the dataset
    look_back = 48  # Use the last 48 hours to predict the next week
    X, y = create_weekly_dataset(scaled_temperature, look_back)

    # Reshape for LSTM input
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Train the model
    model = train_lstm_model(X, y, look_back)

    # Use the most recent data for prediction
    recent_data = scaled_temperature[-look_back:].reshape(1, look_back, 1)
    predicted_temperature = model.predict(recent_data)

    # Invert scaling
    predicted_temperature = scaler.inverse_transform(predicted_temperature)
    return predicted_temperature

def format_predictions(predicted_temperature, start_date):
    """Format predictions into a readable structure."""
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    predictions = []
    for day_offset in range(7):  # Loop through 7 days
        current_date = start_date + datetime.timedelta(days=day_offset)
        day_temps = predicted_temperature[0][day_offset * 24:(day_offset + 1) * 24]

        # Round temperatures to the nearest whole number
        rounded_day_temps = [round(temp) for temp in day_temps]

        # Calculate daily low and high temperatures
        low_temp = float(min(rounded_day_temps))
        high_temp = float(max(rounded_day_temps))   

        predictions.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "low": low_temp,
            "high": high_temp,
            "temperatures": day_temps.tolist()
        })
    return predictions