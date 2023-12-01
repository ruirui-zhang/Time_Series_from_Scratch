import json
import numpy as np
import pandas as pd

def load_model_params():# Load from JSON
    with open('models/model_params.json', 'r') as f:
        loaded_params = json.load(f)
    ar_theta = np.array(loaded_params['ar_theta'])
    ar_intercept = np.array(loaded_params['ar_intercept'])
    ma_theta = np.array(loaded_params['ma_theta'])
    ma_intercept = np.array(loaded_params['ma_intercept'])
    best_p = loaded_params['best_p']
    best_q = loaded_params['best_q']
    trend_model = np.array(loaded_params['trend_model'])
    data = pd.read_csv('data/data_daily.csv')
    return data, ar_theta, ar_intercept, ma_theta, ma_intercept, trend_model, best_p, best_q

def make_prediction(prediction_date, data, ar_theta, ar_intercept, ma_theta, ma_intercept, trend_model, best_p, best_q):
    # Predict the trend
    data['TimeIndex'] = range(len(data))
    data['Date'] = pd.to_datetime(data['# Date'])
    trend = np.polyval(trend_model, data['TimeIndex'])

    # Detrend the data by subtracting the trend component
    detrended = data['Receipt_Count'] - trend
    def apply_ar_model(data, ar_theta, ar_intercept, p):
        ar_forecast = np.zeros(len(data) + 365)
        ar_forecast[:len(data)] = data.copy()
        for i in range(len(data), len(data) + 365):
            ar_forecast[i] = ar_intercept
            for j in range(1, p + 1):
                if i - j >= 0:
                    ar_forecast[i] += ar_theta[j - 1] * ar_forecast[i - j]
        return ar_forecast[len(data):]

    ar_forecast = apply_ar_model(detrended, ar_theta, ar_intercept, best_p)

    # Apply MA model
    def apply_ma_model(data, ma_theta, q):
        ma_forecast = np.zeros(len(data) + 365)
        ma_forecast[:len(data)] = data.copy()
        for i in range(len(data), len(data) + 365):
            for j in range(1, q + 1):
                if i - j >= 0:
                    ma_forecast[i] += ma_theta[j - 1] * ma_forecast[i - j]
        return ma_forecast[len(data):]

    ma_forecast = apply_ma_model(ar_forecast, ma_theta, len(ma_theta))

    # Final forecast (combining AR and MA forecasts)
    final_forecast = ar_forecast + ma_forecast
    forecast_index = range(len(data), len(data) + len(final_forecast))

# Calculate the trend values for the forecast period
    forecast_trend = np.polyval(trend_model, forecast_index)

# Re-add the trend to the forecasted values
    final_forecast = final_forecast + forecast_trend
    last_data_date = list(data['Date'])[-1]
    day_difference = (prediction_date - last_data_date).days
    if 0 <= day_difference < 365:
        # Return the forecasted value for that day
        return final_forecast[day_difference]
    else:
        return None


# Re-add the linear trend to the forecasts (continued from previous code snippet)
# ... [Code for re-adding trend]
