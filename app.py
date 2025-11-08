from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('breast_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    features = [float(x) for x in request.form.values()]
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)[0]
    result = 'Malignant' if prediction == 1 else 'Benign'
    return render_template('index.html', prediction_text=f'Tumor is likely: {result}')

if __name__ == '__main__':
    app.run(debug=True)
