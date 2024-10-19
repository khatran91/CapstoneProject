from flask import Flask, request, jsonify, render_template
import apiModel
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/get_weather', methods=['POST'])
def get_weather():
    data = request.json
    latitude = data['latitude']
    longitude = data['longitude']

    # Call OpenMeteo API to get historical data
    historical_data = apiModel.get_historical_data(latitude, longitude)

    # Predict temperature one year in advance using ARIMA
    prediction = predict_weather(historical_data)

    return jsonify({"predicted_temperature": prediction})

def predict_weather(historical_data):
    # Assuming 'temperature_2m' is the column name in the DataFrame with historical temperature data
    temperature_series = historical_data['temperature_2m']

    # ARIMA model: (p=5, d=1, q=0)
    model = ARIMA(temperature_series, order=(5, 1, 0))
    model_fit = model.fit()

    # Predict the temperature for 365 days in the future
    prediction = model_fit.forecast(steps=365)
    
    # Return the predicted temperature for exactly 1 year from the last date
    return prediction[-1]  # Return the final value of the forecast for the last day

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
