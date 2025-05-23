import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os


def visualize_test_data(data, labels, title="Test Data with Anomalies"):
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
    plt.figure(figsize=(14, 7))

    # Plot the time series data
    plt.plot(data, label="Time Series Data", color='blue', linewidth=1, zorder=1)

    # Detected anomalies
    detected_anomalies = np.where(scores > threshold)[0]
    plt.scatter(detected_anomalies, data[detected_anomalies], color='orange', s=10, label="Detected Anomalies",
                zorder=3)

    # Missed anomalies
    true_anomalies = np.where(labels == 1)[0]
    missed_anomalies = np.setdiff1d(true_anomalies, detected_anomalies)
    if len(missed_anomalies) > 0:
        plt.scatter(missed_anomalies, data[missed_anomalies], color='green', s=15, label="Missed Anomalies", zorder=4)

    # Add darker shaded regions for true anomaly areas
    for start in np.where(np.diff(labels, prepend=0) == 1)[0]:
        end = np.where(np.diff(labels, append=0) == -1)[0]
        end = end[end > start][0] if len(end[end > start]) > 0 else len(labels)
        plt.axvspan(start, end, color='red', alpha=0.3, label="True Anomaly Region", zorder=0)

    # Ensure the "True Anomaly Region" is only in the legend once
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # Add legend and labels
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to a unique file in 'final_results'
    save_folder = "final_results"
    os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist

    # Create a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{title.replace(' ', '_')}_{timestamp}.png"
    filepath = os.path.join(save_folder, filename)

    # Save the figure
    plt.savefig(filepath, dpi=300)
    print(f"Plot saved to: {filepath}")
    plt.show(dpi=300)


def visualize_anomalies(file_path, labels):
    # Load the data
    data = pd.read_csv(file_path, header=None).iloc[:, 0].values

    # Ensure the labels array has the same length as the data
    if len(data) != len(labels):
        raise ValueError("The length of the data and labels must match.")

    # Create a figure
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Time Series Data', color='blue', alpha=0.7)

    # Highlight anomalies
    anomaly_indices = np.where(labels == 1)[0]
    plt.scatter(anomaly_indices, data[anomaly_indices], color='red', label='Anomalies', alpha=0.9)

    # Add titles and labels
    plt.title('Time Series Data with Anomalies')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()
