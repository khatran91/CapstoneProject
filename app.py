from flask import Flask, request, jsonify, render_template
from lstm import predict_weather
import datetime  # Correct import

# Manually define the dates (for testing purposes)
start_date = '2025-11-12'
end_date = '2025-11-19'

# Debugging: Print the available attributes and methods of the datetime module
print("Datetime module content:", dir(datetime))

start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')  # Parse the date
end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')      # Parse the date

print("Start date:", start_date)
print("End date:", end_date)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('dashboard.html')  # Replace 'index.html' with your HTML filename for the map

@app.route('/get_current_weather', methods=['POST'])
def get_current_weather():
    data = request.get_json()  # Assuming you're sending JSON data
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    return jsonify({"weather": "Sunny", "temperature": 75}), 200

@app.route('/get_weather', methods=['POST'])
def get_weather():
    data = request.get_json()  # Get the JSON data from the request
    
    if data:
        start_date = data.get('start_date')  # Get start_date from the incoming data
        end_date = data.get('end_date')      # Get end_date from the incoming data
        
        # Debugging: Print the received start_date and end_date
        print("Start Date:", start_date)
        print("End Date:", end_date)

        # Check if the dates are present and parse them
        if start_date and end_date:
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')  # Parse the date
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')      # Parse the date

            print("Parsed Start Date:", start_date)
            print("Parsed End Date:", end_date)

            # Call the LSTM model's prediction function
            try:
                forecast = predict_weather(start_date, end_date)  # Pass dates to the model
                return jsonify({"forecast": forecast})  # Send the forecast back
            except Exception as e:
                print("Error in prediction:", str(e))
                return jsonify({"error": "Failed to get prediction"}), 500
        else:
            return jsonify({"error": "Start date or end date not provided"}), 400

    return jsonify({"error": "No data received"}), 400  # Handle case with no data

if __name__ == '__main__':
    app.run(debug=True)


## Main one from Kha * Don't touch for now *
# from flask import Flask, request, jsonify, render_template
# import apiModel
# from statsmodels.tsa.arima.model import ARIMA

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('dashboard.html')

# @app.route('/get_weather', methods=['POST'])
# def get_weather():
#     data = request.json
#     latitude = data['latitude']
#     longitude = data['longitude']

#     # Call OpenMeteo API to get historical data
#     historical_data = apiModel.get_historical_data(latitude, longitude)

#     # Predict temperature one year in advance using ARIMA
#     prediction = predict_weather(historical_data)

#     return jsonify({"predicted_temperature": prediction})

# def predict_weather(historical_data):
#     # Assuming 'temperature_2m' is the column name in the DataFrame with historical temperature data
#     temperature_series = historical_data['temperature_2m']

#     # ARIMA model: (p=5, d=1, q=0)
#     model = ARIMA(temperature_series, order=(5, 1, 0))
#     model_fit = model.fit()

#     # Predict the temperature for 365 days in the future
#     prediction = model_fit.forecast(steps=365)
    
#     # Return the predicted temperature for exactly 1 year from the last date
#     return prediction[-1]  # Return the final value of the forecast for the last day

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5001)
