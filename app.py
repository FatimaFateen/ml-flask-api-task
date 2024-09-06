from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get data from POST request
    features = np.array(data["features"]).reshape(1, -1)  # Reshape features to match model input
    prediction = model.predict(features)  # Make prediction
    return jsonify({"prediction": int(prediction[0])})  # Return prediction as JSON

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
