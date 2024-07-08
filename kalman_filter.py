# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_squared_error
# from math import sqrt

# class ExtendedKalmanFilter:
#     def __init__(self, Q, R, initial_state, initial_covariance):
#         self.Q = Q  # Process noise covariance
#         self.R = R  # Measurement noise covariance
#         self.x = initial_state  # Initial state estimate
#         self.P = initial_covariance  # Initial error covariance
        
#     def predict(self):
#         # Predict the next state (since the model is identity, this is simple)
#         self.P = self.P + self.Q

#     def update(self, z):
#         # Kalman Gain
#         K = self.P / (self.P + self.R)
        
#         # Update the state estimate.
#         self.x = self.x + K * (z - self.x)
        
#         # Update the error covariance.
#         self.P = (1 - K) * self.P

#     def get_current_estimate(self):
#         return self.x

# def apply_ekf_to_cgm_data(file_path, Q=0.001, R=1.0, initial_covariance=1.0):
#     # Load the dataset
#     data = pd.read_csv(file_path)
    
#     # Handle NaN values by forward filling
#     data['CGM'].fillna(method='ffill', inplace=True)
    
#     # Extract the CGM data from the dataset
#     cgm_measurements = data['CGM'].values
    
#     # Initialize the EKF with the first CGM reading as the initial state
#     ekf = ExtendedKalmanFilter(Q=Q, R=R, initial_state=cgm_measurements[0], initial_covariance=initial_covariance)
    
#     # Apply EKF to CGM data
#     estimates = []
#     for measurement in cgm_measurements:
#         ekf.predict()
#         ekf.update(measurement)
#         estimates.append(ekf.get_current_estimate())
    
#     # Calculate RMSE between the EKF estimates and actual CGM measurements
#     rmse = sqrt(mean_squared_error(cgm_measurements, estimates))
    
#     return rmse, estimates, cgm_measurements
# def apply_ekf_and_predict_30min_ahead(file_path, Q=0.001, R=1.0, initial_covariance=1.0, projection_steps=6):
#     # Load the dataset
#     data = pd.read_csv(file_path)
    
#     # Handle NaN values by forward filling
#     data['CGM'].fillna(method='ffill', inplace=True)
    
#     # Extract the CGM data from the dataset
#     cgm_measurements = data['CGM'].values
    
#     # Initialize the EKF with the first CGM reading as the initial state
#     ekf = ExtendedKalmanFilter(Q=Q, R=R, initial_state=cgm_measurements[0], initial_covariance=initial_covariance)
    
#     # Apply EKF to CGM data for smoothing
#     estimates = []
#     for measurement in cgm_measurements:
#         ekf.predict()
#         ekf.update(measurement)
#         estimates.append(ekf.get_current_estimate())
    
#     # Predict 30 minutes ahead (6 steps at 5 minutes each)
#     # Simple linear projection based on the last few changes
#     if len(estimates) > projection_steps:
#         last_few_changes = [estimates[-i] - estimates[-(i + 1)] for i in range(1, projection_steps)]
#         avg_change = sum(last_few_changes) / len(last_few_changes)
#         future_value = estimates[-1] + avg_change * projection_steps
#     else:
#         future_value = estimates[-1]  # Not enough data to project
    
#     # Calculate RMSE between the EKF estimates and actual CGM measurements (for smoothing performance)
#     rmse = sqrt(mean_squared_error(cgm_measurements, estimates))
    
#     return rmse, future_value, estimates, cgm_measurements
# # Example usage with the initial data (Replace with the path to your CSV files)
# training_file_path = '570training.csv'  # Replace with your file path
# testing_file_path = '570testing.csv'  # Replace with your file path

# # Apply EKF to training data
# rmse, future_cgm, ekf_estimates, actual_cgm = apply_ekf_and_predict_30min_ahead(training_file_path)


# # Print out the performance metrics
# print("Training RMSE:", rmse)
# print(future_cgm)

# # You might want to plot the results for visual comparison
# import matplotlib.pyplot as plt

# # Plotting the actual versus estimated values for training data
# plt.figure(figsize=(15, 6))
# plt.subplot(1, 2, 1)
# plt.plot(actual_cgm, label='Actual CGM', color='blue')
# plt.plot(ekf_estimates, label='EKF Estimates', color='red')
# plt.title('Training Data: Actual vs EKF Estimates')
# plt.legend()

# # Plotting the actual versus estimated values for testing data
# # plt.subplot(1, 2, 2)
# # plt.plot(actual_testing_cgm, label='Actual CGM', color='blue')
# # plt.plot(testing_estimates, label='EKF Estimates', color='red')
# # plt.title('Testing Data: Actual vs EKF Estimates')
# # plt.legend()

# plt.tight_layout()
# plt.show()


import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from sklearn.metrics import mean_squared_error
training_file_path = '570training.csv'  # Replace with your file path
testing_file_path = '570testing.csv'  
# data = pd.read_csv(file_path)
    
#     # Handle NaN values by forward filling
# data['CGM'].fillna(method='ffill', inplace=True)
    
#     # Extract the CGM data from the dataset
# cgm_measurements = data['CGM'].values

def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)

def prediction_step(X_t, X_t1, A, Q):
    # Predict the state vector X_t1|t
    X_t1_prior = A * X_t

    # Compute the Kalman gain K
    S = Q + A * Q * A.T
    K = Q * A.T * np.linalg.pinv(S)

    # Update the estimate of the state vector
    Y_t1 = normalize_data(X_t1)
    X_t1_posterior = X_t1_prior + K * (Y_t1 - A * X_t1_prior)

    # Compute the new estimate of the error covariance
    KQ = K * Q
    Q_t1 = Q - KQ

    return X_t1_posterior, Q_t1
def run_extended_kalman_filter(data, A, Q, T, num_iterations):
    N = len(data)
    R = np.identity(N) * np.var(data)
    P = np.zeros((N, N))

    # Initial state vector
    X_t = np.zeros((N, 1))

    for _ in range(num_iterations):
        # Update the estimate of the state vector
        X_t_posterior, P = prediction_step(X_t, data, A, P)

        # Predict the next state vector
        X_t = A * X_t_posterior

    return X_t_posterior

def generate_matrix(scale, length):
    A = np.random.normal(scale=scale, size=(length, length))
    return A

# Read the dataset from a file
def read_dataset(file_path):
    data = pd.read_csv(file_path)
    data['CGM'].fillna(method='ffill', inplace=True)
    
    # Extract the CGM data from the dataset
    cgm_measurements = data['CGM'].values
    return cgm_measurements

# Your dataset file paths
train_file_path = '570training.csv'  # Replace with your file path
test_file_path = '570testing.csv'

# Read the dataset
train_data = read_dataset(train_file_path)
test_data = read_dataset(test_file_path)

# Standardize the data
train_data = normalize_data(train_data)
test_data = normalize_data(test_data)

# Define the state transition matrix
A = generate_matrix(scale=0.9, length=len(train_data))

# Define the process noise covariance matrix
Q = np.diag([1e-2] * len(train_data))

# Define the time steps and the number of iterations
T = 1
num_iterations = 100

# Run the extended Kalman filter
predicted_train_data = run_extended_kalman_filter(train_data, A, Q, T, num_iterations)
predicted_test_data = run_extended_kalman_filter(test_data, A, Q, T, num_iterations)

# Concatenate the train and test data
all_data = np.concatenate((train_data, test_data))
predicted_all_data = np.concatenate((predicted_train_data, predicted_test_data))

# Calculate the mean squared error
mse = mean_squared_error(all_data, predicted_all_data)
print('Mean Squared Error:', mse)
