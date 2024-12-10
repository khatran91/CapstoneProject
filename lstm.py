import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
#from datetime import datetime

# Load the weather dataset (skip first 2 rows)
weather = pd.read_csv("weather2024.csv", skiprows=2)

# Display the columns to confirm correct loading
print("Columns in dataset:", weather.columns)

# Check if the dataset contains the 'temperature_2m (°F)' column
if 'temperature_2m (°F)' not in weather.columns:
    raise KeyError("Temperature column not found in the dataset! Please check the column names.")

# Extract the relevant temperature data
temperature_data = weather['temperature_2m (°F)'].values

#######
# Convert 'time' column to datetime
weather['time'] = pd.to_datetime(weather['time'])

# Extract the year and week from the 'time' column
weather['year'] = weather['time'].dt.year
weather['week'] = weather['time'].dt.isocalendar().week

# Calculate the weekly average temperature
weekly_data = weather.groupby(['year', 'week'])['temperature_2m (°F)'].mean().reset_index()

## Next Steps

import datetime

# Function to get the week number and year from a user-input date
def get_week_from_date():
    user_input = input("Enter the date for the week you want to predict (YYYY-MM-DD): ")
    user_date = datetime.datetime.strptime(user_input, "%Y-%m-%d")  # Correct usage of datetime.strptime
    year, week = user_date.isocalendar()[:2]
    print(f"Predicting for year {year}, week {week}.")
    return year, week, user_input


# Get user-specified year and week
target_year, target_week, user_input_date = get_week_from_date()

# Filter data for the same week in previous years
historical_week_data = weather[(weather['week'] == target_week) & (weather['year'] < target_year)]

# Check if there is enough data
if historical_week_data.empty:
    print("No historical data available for the selected week. Please try another week.")
else:
    print(f"Found historical data for {len(historical_week_data['year'].unique())} years.")

# Scale the temperature data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_temperature = scaler.fit_transform(historical_week_data[['temperature_2m (°F)']])

# Create sequences
def create_weekly_dataset(data, look_back=48):
    X, y = [], []
    for i in range(len(data) - look_back - 168):  # 168 hours = 1 week
        X.append(data[i:i + look_back, 0])
        y.append(data[i + look_back: i + look_back + 168, 0])
    return np.array(X), np.array(y)

look_back = 48  # Use the last 48 hours to predict the next week
X, y = create_weekly_dataset(scaled_temperature)

# Reshape for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(look_back, 1)))
model.add(Dense(units=168))  # Predict 168 hours (1 week)

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=20, batch_size=32, verbose=1)

# Use the most recent data from historical_week_data
recent_data = scaled_temperature[-look_back:].reshape(1, look_back, 1)

# Predict next 168 hours
predicted_temperature = model.predict(recent_data)

# Invert scaling
predicted_temperature = scaler.inverse_transform(predicted_temperature)

# Updated function to display predictions
def display_predictions(predicted_temperature, start_date):
    # Parse the start date
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    
    # Loop through the next 7 days, starting from the start date
    print("\nPredicted Hourly Temperatures for the Week:\n")
    
    for day_offset in range(7):  # Loop through each day of the week
        current_date = start_date + datetime.timedelta(days=day_offset)
        
        # Get the weekday name dynamically
        day_of_week = current_date.strftime("%A")  # %A gives the full weekday name (e.g., "Monday")
        
        print(f"{day_of_week} ({current_date.strftime('%Y-%m-%d')}):")
        
        # Loop through 24 hours for each day
        for hour in range(1, 25):  # Loop through 24 hours
            index = day_offset * 24 + (hour - 1)
            temp = predicted_temperature[0][index]
            print(f"  Hour {hour:02}: {temp:.2f}°F")

        print()  # Add a blank line after each day's output
display_predictions(predicted_temperature, user_input_date)