from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model
voting_model = joblib.load(r'C:\Users\WALTON\Documents\GitHub\PD-Detection\voting_ensemble.pkl')

# Create a Flask application
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # Create an index.html file in a templates folder

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json(force=True)
    
    # Assuming the input features are sent as a list
    features = np.array(data['features']).reshape(1, -1)  # Reshape for a single sample
    
    # Make a prediction
    prediction = voting_model.predict(features)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)