from flask import Flask, request, render_template
from prometheus_flask_exporter import PrometheusMetrics
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('healthcare_model.pkl')

metrics = PrometheusMetrics(app)

metrics.info('app_info', 'Healthcare API with Grafana Integration', version='1.0.0')

@app.route('/')
def home():
    return """
        <h1>Welcome to the Healthcare Charge Prediction App!</h1>
        <p>Predict your healthcare charges here:</p>
        <a href="http://127.0.0.1:5001/predict" style="text-decoration: none; background-color: #40916c; color: white; padding: 10px 15px; border-radius: 5px; font-size: 16px;">Start Prediction</a>
    """

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('form.html')
    elif request.method == 'POST':
        try:
           
            features = [
                float(request.form.get('age', 0)),  
                float(request.form.get('bmi', 0)),
                float(request.form.get('children', 0)),
                float(request.form.get('sex_female', 0)),
                float(request.form.get('sex_male', 0)),
                float(request.form.get('smoker_no', 0)),
                float(request.form.get('smoker_yes', 0)),
                float(request.form.get('region_northeast', 0)),
                float(request.form.get('region_northwest', 0)),
                float(request.form.get('region_southeast', 0)),
                float(request.form.get('region_southwest', 0)),
            ]

            features = np.array(features).reshape(1, -1)
            prediction = model.predict(features)
            result = round(float(prediction[0]), 2)

            return render_template('form.html', prediction=result)
        except Exception as e:
            return render_template('form.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=5001)  
