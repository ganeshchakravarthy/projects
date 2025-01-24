from flask import Flask, request, render_template
from prometheus_flask_exporter import PrometheusMetrics
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained ML model
model = joblib.load('healthcare_model.pkl')

# Initialize Prometheus Metrics
metrics = PrometheusMetrics(app)

# Expose default metrics like request count, latency, etc.
metrics.info('app_info', 'Healthcare API with Grafana Integration', version='1.0.0')

@app.route('/')
def home():
    return "Welcome to the Healthcare API with Grafana Integration!"

@app.route('/predict_form', methods=['GET', 'POST'])
def predict_form():
    if request.method == 'GET':
        # Render the form for user inputs
        return render_template('form.html')
    elif request.method == 'POST':
        try:
            # Extract input features from the form
            features = [
                float(request.form.get('age', 0)),  # Default to 0 if missing
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

            # Convert features to NumPy array and reshape
            features = np.array(features).reshape(1, -1)

            # Predict healthcare charges using the ML model
            prediction = model.predict(features)
            result = round(float(prediction[0]), 2)

            # Render the form with the prediction result
            return render_template('form.html', prediction=result)
        except Exception as e:
            return render_template('form.html', error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
