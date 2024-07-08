from math import sqrt
import numpy as np
import pandas as pd
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

def read_and_preprocess_old_data(data_path):
    data = pd.read_csv(data_path)
    data['CGM'].fillna(method='ffill', inplace=True)
    data['basal_insulin'].fillna(method='ffill', inplace=True)
    data['bolus_insulin'].fillna(method='ffill', inplace=True)
    data['CHO'].fillna(method='ffill', inplace=True)
    return data[['CGM', 'basal_insulin', 'bolus_insulin', 'CHO']].values

def fx(x, dt):
    # State transition function
    # Assuming that the basal_insulin, bolus_insulin, and CHO parameters are constant
    return np.array([x[0] + x[1]*dt, x[1], x[2], x[3], x[4]])  # Example transition (CGM, velocity, basal_insulin, bolus_insulin, CHO)

def hx(x):
    # Measurement function
    return np.array([x[0], x[2], x[3], x[4]])  # We measure CGM, basal_insulin, bolus_insulin, CHO

def online_ukf_learning(new_data_stream, old_data):
    points = MerweScaledSigmaPoints(n=5, alpha=0.1, beta=2., kappa=-1)
    ukf = UnscentedKalmanFilter(dim_x=5, dim_z=4, dt=1, fx=fx, hx=hx, points=points)

    ukf.x = np.array([old_data[-1, 0], 0, old_data[-1, 1], old_data[-1, 2], old_data[-1, 3]])  # Initial state
    ukf.P = np.eye(5) * 1000.  # Initial covariance
    ukf.R = np.eye(4)  # Measurement noise
    ukf.Q = np.eye(5) * 0.1  # Process noise

    for measurement in old_data:
        ukf.predict()
        ukf.update(measurement)  # Update with all measurements

    sum_squared_errors = 0
    count = 0

    predictions = []
    for measurement in new_data_stream:
        ukf.predict()
        ukf.update(measurement)  # Update with all measurements

        prediction = ukf.x[0]
        predictions.append(prediction)

        error = measurement[0] - prediction
        sum_squared_errors += error**2
        count += 1
        rmse = sqrt(sum_squared_errors / count)
        print(f"New measurement: {measurement[0]}, Prediction: {prediction}, Updated RMSE: {rmse}")

    return predictions

def read_new_data(data_path):
    data = pd.read_csv(data_path)
    data['CGM'].fillna(method='ffill', inplace=True)
    data['basal_insulin'].fillna(method='ffill', inplace=True)
    data['bolus_insulin'].fillna(method='ffill', inplace=True)
    data['CHO'].fillna(method='ffill', inplace=True)
    return data[['CGM', 'basal_insulin', 'bolus_insulin', 'CHO']].values

def main():
    old_data_path = '570training.csv'
    new_data_path = '540training.csv'
    
    old_data = read_and_preprocess_old_data(old_data_path)
    new_data_stream = read_new_data(new_data_path)

    personalized_predictions = online_ukf_learning(new_data_stream, old_data)

    # Do something with the personalized predictions
    # print(personalized_predictions)

if __name__ == '__main__':
    main()
