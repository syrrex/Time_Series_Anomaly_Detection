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


def create_testfile_with_one_anomaly_each(file_path, anomaly_lengths, gap_between_anomalies=6000):
    if len(anomaly_lengths) != 5:
        raise ValueError("Provide a list of exactly five lengths, one for each type of anomaly.")

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

    # Sequentially inject one anomaly per type
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
    new_file_path = os.path.splitext(file_path)[0] + "_with_anomalies_each.csv"
    pd.DataFrame(data).to_csv(new_file_path, header=False, index=False)

    return new_file_path, labels
