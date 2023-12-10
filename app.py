from flask import Flask, request, render_template, redirect, url_for
import joblib
import numpy as np
import os

# Create a Flask app
app = Flask(__name__)

# Load the trained model and scaler
model_path = os.path.join('path', 'to', 'model.pkl')
scaler_path = os.path.join('path', 'to', 'scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/', methods=['GET'])
def home():
    # Render the home page with the form
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        data = request.form
        try:
            # Prepare data for prediction
            # Prepare data for prediction
            pred_args = [
                int(data.get('AGE') or 0),
                int(data.get('SEX')),
                1 if data.get('PREGNANT') == 'on' else 2,
                int(data.get('MEDICAL_UNIT')),
                int(data.get('PATIENT_TYPE')),
                1 if data.get('USMER') == 'on' else 2,  # Add this line
                1 if data.get('PNEUMONIA') == 'on' else 2,
                1 if data.get('DIABETES') == 'on' else 2,
                1 if data.get('COPD') == 'on' else 2,
                1 if data.get('ASTHMA') == 'on' else 2,
                1 if data.get('INMSUPR') == 'on' else 2,
                1 if data.get('HIPERTENSION') == 'on' else 2,
                1 if data.get('CARDIOVASCULAR') == 'on' else 2,
                1 if data.get('OBESITY') == 'on' else 2,
                1 if data.get('RENAL_CHRONIC') == 'on' else 2,
                1 if data.get('TOBACCO') == 'on' else 2,
                1 if data.get('OTHER_DISEASE') == 'on' else 2,
                int(data.get('CLASIFFICATION_FINAL')),
                int(data.get('COVID_STATUS'))
            ]

            # Convert to numpy array and reshape
            pred_arr = np.array(pred_args)
            pred_arr = scaler.transform(pred_arr.reshape(1, -1))

            # Make prediction
            prediction = model.predict(pred_arr)
            prediction_proba = model.predict_proba(pred_arr)
            death_risk_percentage = prediction_proba[0][1] * 100  # Convert to percentage

            # Return prediction result page
            return render_template('predict.html', prediction=death_risk_percentage)
        except Exception as e:
            return str(e)
    else:
        # Render the prediction form page
        return render_template('predict.html', prediction=None)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
