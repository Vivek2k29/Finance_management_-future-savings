# app.py
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load your pre-trained model
model = joblib.load('savings_model.pkl')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [
        data['current_income'],
        data['monthly_expenses'],
        data['current_savings'],
        data['loan_amount'],
        data['credit_score']
    ]

    # Make prediction
    future_savings = model.predict([features])[0]  # Adjust based on your model's input
    return jsonify({'future_savings': future_savings})


if __name__ == '__main__':
    app.run(debug=True)

