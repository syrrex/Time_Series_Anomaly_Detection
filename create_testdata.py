import numpy as np
import os
from plots import *
import pandas as pd


def create_testfile_with_sequential_anomalies(
        file_path,
        anomaly_lengths,
        gap_between_anomalies=5000
):
    """
    Creates a test file with sequential anomalies of varying lengths and returns the file path and anomaly labels.

    Parameters:
        file_path (str): Path to the input CSV file containing time series data.
        anomaly_lengths (list): List of lengths for each anomaly type.
        gap_between_anomalies (int): Number of normal data points between two anomalies. Default is 5000.

    Returns:
        new_file_path (str): Path to the new CSV file with injected anomalies.
        labels (numpy array): Array with 0 for normal and 1 for anomalous data points.
    """
    if len(anomaly_lengths) != 6:
        raise ValueError("Provide a list of six lengths for the anomalies, one for each type.")

    # Load the original data from the CSV file
    data = pd.read_csv(file_path, header=None).iloc[:, 0].values

    # Extend the data by appending itself to ensure sufficient space for anomalies
    data = np.concatenate([data, data])

    # Convert data to float to prevent casting errors during anomaly injection
    data = data.astype(np.float64)

    # Initialize labels array with zeros
    labels = np.zeros(len(data), dtype=int)

    # Define anomaly types and corresponding injection methods
    anomaly_types = ['constant', 'peak', 'trough', 'reverse', 'noise']
    std_dev = np.std(data)
    mean_value = np.mean(data)

    # Sequentially inject anomalies into the data
    current_position = 2000
    for anomaly_type, anomaly_length in zip(anomaly_types, anomaly_lengths):
        if current_position + anomaly_length > len(data):
            break  # Stop if there's not enough space for the next anomaly

        start = current_position
        end = start + anomaly_length

        # Inject the anomaly
        if anomaly_type == 'constant':
            data[start:end] = mean_value
        elif anomaly_type == 'peak':
            data[start:end] += std_dev * 10
        elif anomaly_type == 'trough':
            data[start:end] -= std_dev * 10
        elif anomaly_type == 'reverse':
            data[start:end] = data[start:end][::-1]
        elif anomaly_type == 'noise':
            data[start:end] += np.random.normal(0, std_dev * 3, size=anomaly_length)
        elif anomaly_type == 'trend':
            trend = np.linspace(0, std_dev * 1.5, anomaly_length)
            data[start:end] += trend

        # Update labels for the anomalous segment
        labels[start:end] = 1

        # Move the current position forward by the anomaly length and the gap
        current_position = end + gap_between_anomalies

    # Trim data and labels to the original length
    data = data[:len(data) // 2]
    labels = labels[:len(data)]

    # Save the modified data to a new CSV file
    new_file_path = os.path.splitext(file_path)[0] + "_with_sequential_anomalies.csv"
    pd.DataFrame(data).to_csv(new_file_path, header=False, index=False)

    return new_file_path, labels


if __name__ == '__main__':
    file_path = "X_train.csv"
    # ['constant', 'peak', 'trough', 'reverse', 'noise', 'trend']
    anomaly_lengths = [350, 200, 200, 1000, 500, 5000]
    gap_between_anomalies = 5000

    new_file_path, labels = create_testfile_with_sequential_anomalies(
        file_path,
        anomaly_lengths=anomaly_lengths,
        gap_between_anomalies=gap_between_anomalies
    )

    print(f"New file created at: {new_file_path}")

    visualize_anomalies(new_file_path, labels)
