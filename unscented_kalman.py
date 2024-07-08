from math import sqrt
import numpy as np
import pandas as pd
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

def read_and_preprocess_old_data(data_path):
    # Same as before
    data = pd.read_csv(data_path)
    data['CGM'].fillna(method='ffill', inplace=True)
    cgm_measurements = data['CGM'].values
    return cgm_measurements

def fx(x, dt):
    # State transition function
    return np.array([x[0] + x[1]*dt, x[1]])  # Example transition (position, velocity)

def hx(x):
    # Measurement function
    return np.array([x[0]])  # Direct observation of the first state variable

def online_ukf_learning(new_data_stream, old_data):
    points = MerweScaledSigmaPoints(n=2, alpha=0.1, beta=2., kappa=-1)
    ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1, dt=1, fx=fx, hx=hx, points=points)

    ukf.x = np.array([old_data[-1], 0])  # Initial state
    ukf.P *= 1000.  # Initial covariance
    ukf.R = np.diag([1])  # Measurement noise
    ukf.Q = Q_discrete_white_noise(dim=2, dt=1, var=0.1)  # Process noise

    for measurement in old_data:
        ukf.predict()
        ukf.update(measurement)

    sum_squared_errors = 0
    count = 0

    predictions = []
    for measurement in new_data_stream:
        ukf.predict()
        ukf.update(measurement)
        
        prediction = ukf.x[0]
        predictions.append(prediction)

        error = measurement - prediction
        sum_squared_errors += error**2
        count += 1
        rmse = sqrt(sum_squared_errors / count)
        print(f"New measurement: {measurement}, Prediction: {prediction}, Updated RMSE: {rmse}")

    return predictions

def read_new_data(data_path):
    # Same as before
    data = pd.read_csv(data_path)
    data['CGM'].fillna(method='ffill', inplace=True)
    cgm_measurements = data['CGM'].values
    return cgm_measurements

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
