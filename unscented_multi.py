import numpy as np
import pandas as pd
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from sklearn.preprocessing import MinMaxScaler

def read_and_preprocess_data(data_path):
    data = pd.read_csv(data_path)
    data[['CGM', 'basal_insulin', 'bolus_insulin', 'CHO']] = data[['CGM', 'basal_insulin', 'bolus_insulin', 'CHO']].fillna(method='ffill').fillna(method='bfill')
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0)
    return data[['CGM', 'basal_insulin', 'bolus_insulin', 'CHO']]

def normalize_data(data):
    scalers = {}
    normalized_data = pd.DataFrame(index=data.index)
    for column in data.columns:
        scalers[column] = MinMaxScaler()
        normalized_data[column] = scalers[column].fit_transform(data[[column]]).flatten()
    return normalized_data, scalers

def fx(x, dt):
    # State transition function
    # x = [CGM, CGM_rate, Basal, Bolus, CHO]
    cgm_next = x[0] + x[1] * dt
    cgm_rate_next = x[1] - 0.01 * x[2] - 0.02 * x[3] + 0.01 * x[4]
    return np.array([cgm_next, cgm_rate_next, x[2], x[3], x[4]])

def hx(x):
    # Measurement function - we only measure CGM
    return np.array([x[0]])

def online_ukf_learning(new_data_stream, old_data, cgm_scaler, window_size=100):
    n = 5  # State dimension
    m = 1  # Measurement dimension (only CGM)
    dt = 5  # Time step (assuming 5 minutes between measurements)
    
    points = MerweScaledSigmaPoints(n=n, alpha=0.1, beta=2., kappa=-1)
    ukf = UnscentedKalmanFilter(dim_x=n, dim_z=m, dt=dt, fx=fx, hx=hx, points=points)
    
    # Initialize state
    ukf.x = np.array([old_data['CGM'].iloc[-1], 0, old_data['basal_insulin'].iloc[-1], old_data['bolus_insulin'].iloc[-1], old_data['CHO'].iloc[-1]])
    ukf.P = np.eye(n) * 100
    ukf.R = np.array([[0.1]])  # Measurement noise (1x1 matrix for CGM only)
    ukf.Q = np.eye(n) * 0.01
    
    training_window = old_data[-window_size:]
    for _, measurement in training_window.iterrows():
        ukf.predict()
        ukf.update(measurement['CGM'])  # Only pass CGM value
    
    predictions = []
    actual_measurements = []
    
    for _, measurement in new_data_stream.iterrows():
        # Update the input variables in the state
        ukf.x[2:] = [measurement['basal_insulin'], measurement['bolus_insulin'], measurement['CHO']]
        
        ukf.predict()
        prediction = ukf.x[0]
        
        ukf.update(measurement['CGM'])  # Only pass CGM value
        
        predictions.append(prediction)
        actual_measurements.append(measurement['CGM'])
    
    predictions = np.array(predictions).reshape(-1, 1)
    actual_measurements = np.array(actual_measurements).reshape(-1, 1)

    # Inverse transform
    predictions_original = cgm_scaler.inverse_transform(predictions)[:, 0]
    actual_measurements_original = cgm_scaler.inverse_transform(actual_measurements)[:, 0]

    # Calculate RMSE in original scale
    errors = predictions_original - actual_measurements_original
    rmse = np.sqrt(np.mean(errors**2))
    
    for pred, actual, error in zip(predictions_original[:10], actual_measurements_original[:10], errors[:10]):
        print(f"Prediction: {pred:.2f}, Actual: {actual:.2f}, Error: {error:.2f}")
    
    return predictions_original, rmse

def main():
    old_data_path = '570training.csv'
    new_data_path = '540training.csv'
    
    old_data = read_and_preprocess_data(old_data_path)
    new_data = read_and_preprocess_data(new_data_path)
    
    old_data_normalized, scalers = normalize_data(old_data)
    new_data_normalized = pd.DataFrame(index=new_data.index)
    for column in new_data.columns:
        new_data_normalized[column] = scalers[column].transform(new_data[[column]]).flatten()
    
    personalized_predictions, final_rmse = online_ukf_learning(new_data_normalized, old_data_normalized, scalers['CGM'])
    
    print(f"Final RMSE: {final_rmse:.2f}")

if __name__ == '__main__':
    main()