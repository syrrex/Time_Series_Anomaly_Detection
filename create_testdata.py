import numpy as np
import os
from plots import *
import pandas as pd


def create_testfile_with_sequential_anomalies(
        file_path,
        anomaly_lengths,
        gap_between_anomalies=5000
):
    if len(anomaly_lengths) != 5 or not all(len(lengths) == 3 for lengths in anomaly_lengths):
        raise ValueError("Provide a list of five sublists, each containing three lengths for the anomalies.")

    # Load the original data from the CSV file
    data = pd.read_csv(file_path, header=None).iloc[:, 0].values

    # Extend the data by appending itself to ensure sufficient space for anomalies
    data = np.concatenate([data, data, data])

    # Convert data to float to prevent casting errors during anomaly injection
    data = data.astype(np.float64)

    # Initialize labels array with zeros
    labels = np.zeros(len(data), dtype=int)

    # Define anomaly types and corresponding injection methods
    anomaly_types = ['constant', 'peak', 'trough', 'reverse', 'noise']
    std_dev = np.std(data)
    mean_value = np.mean(data)

    # Sequentially inject anomalies into the data (3 anomalies per type, with variable lengths)
    current_position = 2000
    for anomaly_type, anomaly_lengths_for_type in zip(anomaly_types, anomaly_lengths):
        for anomaly_length in anomaly_lengths_for_type:  # Iterate through the 3 lengths for this type
            if current_position + anomaly_length > len(data):
                break  # Stop if there's not enough space for the next anomaly

            start = current_position
            end = start + anomaly_length

            # Inject the anomaly
            if anomaly_type == 'constant':
                data[start:end] = mean_value
            elif anomaly_type == 'peak':
                data[start:end] += std_dev * 5
            elif anomaly_type == 'trough':
                data[start:end] -= std_dev * 5
            elif anomaly_type == 'reverse':
                data[start:end] = data[start:end][::-1]
            elif anomaly_type == 'noise':
                data[start:end] += np.random.normal(0, std_dev * 2, size=anomaly_length)

            # Update labels for the anomalous segment
            labels[start:end] = 1

            # Move the current position forward by the anomaly length and the gap
            current_position = end + gap_between_anomalies

    # Trim data and labels to the original length (first third of the extended data)
    data = data[:len(data) // 3]
    labels = labels[:len(data)]

    # Save the modified data to a new CSV file
    new_file_path = os.path.splitext(file_path)[0] + "_with_sequential_anomalies.csv"
    pd.DataFrame(data).to_csv(new_file_path, header=False, index=False)

    return new_file_path, labels



if __name__ == '__main__':
    file_path = "X_train.csv"
    # ['constant', 'peak', 'trough', 'reverse', 'noise', 'trend']
    anomaly_lengths = [
        [50, 250, 500],  # Lengths for 'constant' anomalies
        [10, 100, 250],  # Lengths for 'peak' anomalies
        [10, 100, 250],  # Lengths for 'trough' anomalies
        [100, 50, 500],  # Lengths for 'reverse' anomalies
        [80, 100, 300]  # Lengths for 'noise' anomalies
    ]
    gap_between_anomalies = 2000

    new_file_path, labels = create_testfile_with_sequential_anomalies(
        file_path,
        anomaly_lengths=anomaly_lengths,
        gap_between_anomalies=gap_between_anomalies
    )

    print(f"New file created at: {new_file_path}")

    visualize_anomalies(new_file_path, labels)
