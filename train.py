from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import joblib

file_path = 'X_train.csv'


def differencing_data(data, order=1):
    differenced_data = np.diff(data, n=order)
    differenced_data = np.insert(differenced_data, 0, 0)
    return differenced_data


def log_transform_data(data):
    return np.log1p(np.abs(data))


def reverse_differencing(differenced_data, initial_value):
    return np.r_[initial_value, differenced_data].cumsum()


def train_model(time_series, window_size=10, k=5, distance_metric='euclidean', differencing=False, log_transform=False):
    # Normalize the data
    scaler = MinMaxScaler()

    # Remove trend
    if differencing:
        print("Differencing the data...")
        time_series = differencing_data(time_series)

    if log_transform:
        print("Log transforming the data...")
        time_series = log_transform_data(time_series)

    # scale the data
    time_series = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()

    # Create sliding windows
    windows = []
    for i in range(len(time_series) - window_size + 1):
        windows.append(time_series[i:i + window_size])
    windows = np.array(windows)

    # Train KNN model
    knn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric=distance_metric)
    knn.fit(windows)

    # Compute distances to k-nearest neighbors
    distances, _ = knn.kneighbors(windows)
    baseline_distances = np.mean(distances, axis=1)

    return knn, baseline_distances, scaler


if __name__ == '__main__':
    # Path to your CSV file
    file_path = 'X_train.csv'

    # Load the time series data
    time_series = pd.read_csv(file_path, header=None).values.flatten()

    # Parameters for the most stable model found in with grid search
    #window_size = 250
    #k = 2
    #distance_metric = 'manhattan'

    window_size = 250
    k = 2
    distance_metric = 'manhattan'


    # Train the K-Nearest Windows model
    knn_model, baseline_distances, scaler = train_model(
        time_series,
        window_size=window_size,
        k=k,
        distance_metric=distance_metric,
        log_transform=True
    )

    # Output results
    print("Model trained successfully!")
    # Save the model and scaler
    model_path = 'knn_model.joblib'
    scaler_path = 'scaler.joblib'
    baseline_path = 'baseline_distances.npy'

    # Save the KNN model
    joblib.dump(knn_model, model_path)
    print(f"Model saved to {model_path}")

    # Save the scaler
    if scaler:
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")
