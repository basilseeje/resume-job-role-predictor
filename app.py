from flask import Flask, render_template, request
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load model and tools
model = load_model("resume_model.keras")
vectorizer = joblib.load("vectorizer.pkl")
encoder = joblib.load("encoder.pkl")

# Create app
app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    skills = request.form['skills']

    # Convert input
    X = vectorizer.transform([skills])
    X = X.toarray().astype('float32')

    # Predict
    prediction = model.predict(X)
    predicted_class = np.argmax(prediction, axis=1)

    # Convert to job role
    job_role = encoder.inverse_transform(predicted_class)[0]

    return render_template('index.html', prediction_text=f'Predicted Role: {job_role}')

# Run app
if __name__ == "__main__":
    app.run(debug=True)
    