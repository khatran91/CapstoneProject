import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Load the dataset
weather = pd.read_csv("weather2024.csv", skiprows=2)

# Filter out for just temperature data
temperature_data = weather[['time', 'temperature_2m (°F)']]

# Attempts to convert 'time' to datetime, coerce errors to NaT, and drop rows with NaT in 'time'
temperature_data.loc[:, 'time'] = pd.to_datetime(temperature_data['time'], errors='coerce')
temperature_data = temperature_data.dropna(subset=['time'])

# Converts 'time' to timestamp (numeric format)
temperature_data['time'] = temperature_data['time'].apply(lambda x: x.timestamp() if pd.notna(x) else np.nan)

# Define X and y
X = temperature_data[['time']].values
y = temperature_data['temperature_2m (°F)'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# If you want to work with pandas methods:
X_train_df = pd.DataFrame(X_train)

# Now you can use isna() on X_train_df
#print(X_train_df.isna().sum())

# Print the type of y_train
#print("Type of y_train:", type(y_train))

# Print a few values of y_train to inspect its contents
#print("First few values of y_train:", y_train[:10])

# Convert y_train to a pandas Series first to use pd.to_numeric
y_train = pd.to_numeric(y_train, errors='coerce')  # 'coerce' will turn invalid values into NaN

# Check the conversion result
#print("NaNs in y_train after conversion:", pd.isna(y_train).sum())

# Identify non-NaN indices for y_train
mask_y_train = ~np.isnan(y_train)

# Apply the mask to both X_train and y_train
X_train_clean = X_train[mask_y_train]
y_train_clean = y_train[mask_y_train]

# Verify no NaNs are present
#print("NaNs in X_train after cleaning:", np.isnan(X_train_clean).sum())
#print("NaNs in y_train after cleaning:", np.isnan(y_train_clean).sum())

# Initialize the model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train_clean, y_train_clean)

# Make predictions on the training data
y_train_pred = model.predict(X_train_clean)

# Evaluate the model performance on training data
mse_train = mean_squared_error(y_train_clean, y_train_pred)  # Mean Squared Error
r2_train = r2_score(y_train_clean, y_train_pred)  # R-squared value

# Print out the evaluation metrics for training data
#print(f"Training Mean Squared Error (MSE): {mse_train}")
#print(f"Training R-squared (R²): {r2_train}")

# Now, for testing data

# Ensure y_test is a numeric type, coercing any non-numeric values into NaN
y_test = pd.to_numeric(y_test, errors='coerce')

# Check for NaNs in y_test
#print("NaNs in y_test:", np.isnan(y_test).sum())

# Clean the data by removing NaNs from y_test
mask_test = ~np.isnan(y_test)  # Mask to remove NaNs
y_test_clean = y_test[mask_test]
X_test_clean = X_test[mask_test]  # Apply the same mask to X_test

# Make predictions on the cleaned test data
y_test_pred = model.predict(X_test_clean)

# Evaluate the model performance on test data
mse_test = mean_squared_error(y_test_clean, y_test_pred)  # Mean Squared Error
r2_test = r2_score(y_test_clean, y_test_pred)  # R-squared value

# Print out the evaluation metrics for test data
print(f"Test Mean Squared Error (MSE): {mse_test}")
print(f"Test R-squared (R²): {r2_test}")



# //// Plotting /////

# Plot the training data and the regression line
plt.figure(figsize=(12, 6))

# Plot the training data
plt.scatter(X_train_clean, y_train_clean, color='blue', label='Training data')

# Plot the predicted values on the training data
plt.plot(X_train_clean, y_train_pred, color='red', label='Regression line (Train)')

# Plot the testing data
plt.scatter(X_test_clean, y_test_clean, color='green', label='Test data')

# Plot the predicted values on the test data
plt.plot(X_test_clean, y_test_pred, color='orange', label='Regression line (Test)', linestyle='--')

# Adding labels and title
plt.xlabel('Time (timestamp)')
plt.ylabel('Temperature (°F)')
plt.title('Linear Regression Model: Temperature vs Time')

# Show legend
plt.legend()

# Display the plot
plt.show()

# Function to get a prediction based on a user-entered date
def predict_temperature():
    # Ask the user to input a date
    user_input = input("Enter a date (YYYY-MM-DD HH:MM:SS): ")
    
    try:
        # Convert the input string to a datetime object
        user_date = datetime.strptime(user_input, "%Y-%m-%d %H:%M:%S")
        
        # Convert the datetime object to a timestamp
        user_timestamp = user_date.timestamp()
        
        # Reshape for prediction
        user_input_data = np.array([[user_timestamp]])
        
        # Make the prediction
        predicted_temp = model.predict(user_input_data)
        
        # Display the result
        print(f"The predicted temperature on {user_date} is {predicted_temp[0]:.2f}°F")
    
    except ValueError:
        print("Invalid date format. Please use the format YYYY-MM-DD HH:MM:SS")

# Calls the function
# predict_temperature()

print(weather.data)