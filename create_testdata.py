import numpy as np
import pandas as pd
import os

# TODO: Create a reasonable amount of reasonable anomalies to test the model

def create_testdata(file_path, anomaly_length=350):
    """
    Reads time series data from a file, injects an anomaly sequence of specified length,
    saves the modified data to a new file with only one column, and returns the new file path and labels.

    Parameters:
        file_path (str): Path to the input CSV file containing time series data.
        anomaly_length (int): Length of the anomaly sequence to inject. Default is 350.

    Returns:
        new_file_path (str): Path to the new CSV file with injected anomalies.
        labels (numpy array): Array with 0 for normal and 1 for anomalous data points.
    """
    # Load the original data from the CSV file
    data = pd.read_csv(file_path, header=None).iloc[:, 0].values

    # Convert data to float to prevent casting errors during anomaly injection
    data = data.astype(np.float64)

    # Initialize labels array with zeros
    labels = np.zeros(len(data), dtype=int)

    # Ensure the anomaly length is less than the length of the data
    if anomaly_length >= len(data):
        raise ValueError("Anomaly length must be less than the length of the data.")

    # Randomly choose a start position for the anomaly
    start = np.random.randint(0, len(data) - anomaly_length)
    end = start + anomaly_length

    # Inject an anomaly by adding a significant deviation
    anomaly_magnitude = 15 * np.std(data)  # Adjust the multiplier as needed
    data[start:end] += anomaly_magnitude

    # Update labels for the anomalous segment
    labels[start:end] = 1

    # Generate a new file path
    base, ext = os.path.splitext(file_path)
    new_file_path = f"{base}_with_anomaly{ext}"

    # Save only the modified data to the new CSV file (no header)
    pd.DataFrame(data).to_csv(new_file_path, index=False, header=False)

    return new_file_path, labels
