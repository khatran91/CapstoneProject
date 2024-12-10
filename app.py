from flask import Flask, render_template, request, jsonify

from lstm import display_predictions  # Import your LSTM model prediction function

 

app = Flask(__name__)

 

@app.route('/')

def index():

    return render_template('dashboard.html')  # Ensure 'dashboard.html' is in the 'templates' folder

 

@app.route('/submit', methods=['POST'])

def submit():

    # Handle submitted data

    data = request.get_json()  # Get data from the form

    return jsonify({"message": "Data received"}), 200

 

# @app.route('/')

# def index():

#     return app.send_static_file('dashboard.html')  # Serve your HTML file

# @app.route('/predict', methods=['POST'])

# def predict():

#     data = request.get_json()

#     latitude = data.get('latitude')

#     longitude = data.get('longitude')

#     start_date = data.get('start_date')

#     end_date = data.get('end_date')

 

#     predictions = predict_weather(latitude, longitude, start_date, end_date)

#     return jsonify(predictions)

 

if __name__ == "__main__":

    app.run(debug=True)

 