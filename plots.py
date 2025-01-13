import matplotlib.pyplot as plt
import numpy as np

def visualize_test_data(data, labels, title="Test Data with Anomalies"):
    """
    Visualize the test data with anomalies highlighted.

    Parameters:
        data (numpy array): Time series data.
        labels (numpy array): Labels indicating anomalies (1 for anomaly, 0 for normal).
        title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Time Series Data", color='blue')

    # Highlight anomalies
    anomalies = np.where(labels == 1)[0]
    plt.scatter(anomalies, data[anomalies], color='red', label="Anomalies", zorder=3)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()


def visualize_detected_anomalies(data, labels, scores, threshold=0.5, title="Detected Anomalies"):
    """
    Visualize the detected anomalies in the test data.

    Parameters:
        data (numpy array): Time series data.
        labels (numpy array): Ground truth labels (1 for anomaly, 0 for normal).
        scores (numpy array): Anomaly scores (0-1, higher = more anomalous).
        threshold (float): Threshold to classify anomalies based on scores.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Time Series Data", color='blue')

    # Detected anomalies (predicted anomalies)
    detected_anomalies = np.where(scores > threshold)[0]
    plt.scatter(detected_anomalies, data[detected_anomalies], color='orange', label="Detected Anomalies", zorder=3)

    # True anomalies (from ground truth)
    true_anomalies = np.where(labels == 1)[0]
    plt.scatter(true_anomalies, data[true_anomalies], color='red', label="True Anomalies", zorder=4)

    # Missed anomalies
    missed_anomalies = np.setdiff1d(true_anomalies, detected_anomalies)
    if len(missed_anomalies) > 0:
        plt.scatter(missed_anomalies, data[missed_anomalies], color='purple', label="Missed Anomalies", zorder=5)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()
