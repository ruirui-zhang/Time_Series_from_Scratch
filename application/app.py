from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import datetime
from model import load_model_params, make_prediction

# Create a Flask app
app = Flask(__name__)


# Route for the form
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get date from the form
        date_input = request.form['date']
        
        prediction_date = pd.to_datetime(date_input)
        
        # Call your prediction function
        data, ar_theta, ar_intercept, ma_theta, ma_intercept, trend_model, best_p, best_q = load_model_params()
        prediction = make_prediction(prediction_date, data, ar_theta, ar_intercept, ma_theta, ma_intercept, trend_model, best_p, best_q)
        
        # Render the result template with the prediction and the input date
        return render_template('result.html', prediction=prediction, date=prediction_date.strftime('%B %Y'))
    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
