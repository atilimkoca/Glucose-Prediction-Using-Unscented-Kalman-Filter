# from math import sqrt
# import numpy as np
# import pandas as pd
# from filterpy.kalman import KalmanFilter
# from filterpy.common import Q_discrete_white_noise
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import StandardScaler

# def read_and_preprocess(data_path):
#     # Implement your data reading and preprocessing here
#     # ...
#     data = pd.read_csv(data_path)
#     data['CGM'].fillna(method='ffill', inplace=True)
    
#     # Extract the CGM data from the dataset
#     cgm_measurements = data['CGM'].values
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(cgm_measurements.reshape(-1, 1))
#     return scaled_data, scaler

# def implement_ekf(scaled_data):
#     n = 2  # Adjusted dimension, e.g., [glucose level, rate of change]
#     kf = KalmanFilter(dim_x=n, dim_z=1)  # Observation space remains 1-dimensional

#     # Initial state
#     kf.x = np.zeros(n)  # initial state - position and velocity

#     # State Transition Matrix (F)
#     # Assuming the first state is the value and the second is its rate of change
#     kf.F = np.array([[1, 1],  # [1, dt] if time step dt is other than 1
#                      [0, 1]])

#     # Measurement Function (H)
#     kf.H = np.array([[1, 0]])  # We only measure the first variable (glucose level)

#     # ... (rest of your code for covariance matrices and noise)

#     # Implementing EKF loop as before
#     predictions = []
#     for measurement in scaled_data:
#         kf.predict()
#         kf.update([measurement])  # Update with the latest observation
#         predictions.append(kf.x[0])  # Assuming first state variable is what we want

#     return np.array(predictions)
# def calculate_rmse(actual, predicted):
#     return sqrt(mean_squared_error(actual, predicted))
# def main():
#     # Read and preprocess data
#     data_path = '570training.csv'
#     scaled_data, scaler = read_and_preprocess(data_path)

#     # Implement EKF
#     predictions = implement_ekf(scaled_data)

#     # Inverse transform to original scale if data was scaled
#     inv_predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
#     actual = scaler.inverse_transform(scaled_data.reshape(-1, 1)).flatten()  # Assuming you have the actual values in similar shape/format
#     print(actual, inv_predictions)
#     # Calculate RMSE
#     rmse = calculate_rmse(actual, inv_predictions)
#     print(f"RMSE: {rmse}")

# if __name__ == '__main__':
#     main()


from math import sqrt
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import pandas as pd
from sklearn.preprocessing import StandardScaler

def read_and_preprocess_old_data(data_path):
    # Read historical data and preprocess it
    # This could be similar to your previous data loading method
    # ...
    data = pd.read_csv(data_path)
    data['CGM'].fillna(method='ffill', inplace=True)
    
    # Extract the CGM data from the dataset
    cgm_measurements = data['CGM'].values
    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(cgm_measurements.reshape(-1, 1))
    return cgm_measurements

def online_ekf_learning(new_data_stream, old_data):
    n = 2  # Dimension of the state
    kf = KalmanFilter(dim_x=n, dim_z=1)

    # Initialize filter using old data
    kf.x = np.array([old_data[-1], 0])  # Last known measurement and an assumed rate of change
    kf.F = np.array([[1, 1], [0, 1]])  # State transition matrix
    kf.H = np.array([[1, 0]])  # Observation matrix
    kf.P *= 1000.  # Initial covariance matrix
    kf.Q = Q_discrete_white_noise(dim=2, dt=1, var=0.1)  # Process noise
    kf.R = np.array([[1]])  # Observation noise

    # Prime the filter with old data if needed
    for measurement in old_data:
        kf.predict()
        kf.update([measurement])

    # Initialize variables for RMSE calculation
    sum_squared_errors = 0
    count = 0

    # Now process new data
    predictions = []
    for measurement in new_data_stream:
        kf.predict()
        kf.update([measurement])
        
        # Update predictions
        prediction = kf.x[0]
        predictions.append(prediction)

        # Update RMSE calculation
        error = measurement - prediction
        sum_squared_errors += error**2
        count += 1
        rmse = sqrt(sum_squared_errors / count)
        print(f"New measurement: {measurement}, Prediction: {prediction}, Updated RMSE: {rmse}")

    return predictions
def read_new_data(data_path):
    # Implement your data reading here. It should return data as a stream (one at a time)
    # ...
    data = pd.read_csv(data_path)
    data['CGM'].fillna(method='ffill', inplace=True)
    
    # Extract the CGM data from the dataset
    cgm_measurements = data['CGM'].values
    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(cgm_measurements.reshape(-1, 1))
    return cgm_measurements
def main():
    old_data_path = '570training.csv'
    new_data_path = '540training.csv'
    
    old_data = read_and_preprocess_old_data(old_data_path)
    new_data_stream = read_new_data(new_data_path)  # Should be a generator or iterator

    # Implement Online EKF Learning
    personalized_predictions = online_ekf_learning(new_data_stream, old_data)

    # Do something with the personalized predictions
    #print(personalized_predictions)

if __name__ == '__main__':
    main()
