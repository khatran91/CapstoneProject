from flask import Flask, request, jsonify, render_template
import lstm  # Import the refactored LSTM module

app = Flask(__name__)

@app.route('/')
def home():
    """Render the dashboard for user interaction."""
    return render_template('dashboard.html')

@app.route('/get_weather', methods=['POST'])
def get_weather():
    """
    API endpoint to get weather predictions based on user input.
    Receives JSON payload with 'city' and 'start_date'.
    """
    try:
        # Parse user inputs from the request
        data = request.json
        city = data['city']
        start_date = data['start_date']

        # Load city-specific weather data
        weather_data = lstm.load_city_data(city)

        # Predict weather using LSTM
        predicted_temperature = lstm.predict_weather(weather_data, start_date)

        # Format predictions for frontend display
        formatted_predictions = lstm.format_predictions(predicted_temperature, start_date)

        # Return the predictions as JSON
        return jsonify({
            "success": True,
            "predictions": formatted_predictions
        })

    except FileNotFoundError:
        return jsonify({"success": False, "error": f"Data for {city} not found. Please try another city."}), 404

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400

    except Exception as e:
        return jsonify({"success": False, "error": "An unexpected error occurred.", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)