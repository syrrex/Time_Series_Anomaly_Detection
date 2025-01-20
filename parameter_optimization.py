from test import *
import os
import pandas as pd
from train import train_model
from create_testdata import create_testfile_with_sequential_anomalies
from plots import visualize_detected_anomalies
from pate.PATE_metric import PATE
import matplotlib.pyplot as plt


def test_parameters(time_series, test_file, labels, parameter_grid, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through the parameter grid
    for window_size in parameter_grid["window_size"]:
        for k in parameter_grid["k"]:
            for threshold_factor in parameter_grid["threshold_factor"]:
                for distance_metric in parameter_grid["distance_metric"]:
                    print(
                        f"Testing: window_size={window_size}, k={k}, threshold_factor={threshold_factor}, distance_metric={distance_metric}")

                    # Train the model using the training data
                    knn_model, _, scaler = train_model(
                        time_series,
                        window_size=window_size,
                        k=k,
                        distance_metric=distance_metric
                    )

                    # Apply anomaly detection on the test data
                    scores = apply_anomaly_detection(test_file, knn_model, scaler, window_size, threshold_factor)

                    # Compute PATE metric
                    pate_score = PATE(labels, scores, binary_scores=False)
                    print(f"PATE Score: {pate_score:.4f}")

                    # Save plot
                    plot_title = f"ws={window_size}_k={k}_th={threshold_factor}_dm={distance_metric}_pate={pate_score:.4f}"
                    plot_filename = os.path.join(output_dir, f"{plot_title}.png")
                    visualize_detected_anomalies(
                        data=pd.read_csv(test_file, header=None).iloc[:, 0].values,
                        labels=labels,
                        scores=scores,
                        threshold=0.5,
                        title=plot_title
                    )
                    plt.savefig(plot_filename)
                    plt.clf()
                    plt.close()
                    print(f"Plot saved: {plot_filename}")

                    # Save scores and labels
                    results_filename = os.path.join(output_dir, f"{plot_title}_scores.csv")
                    results_df = pd.DataFrame({"Labels": labels, "Scores": scores})
                    results_df.to_csv(results_filename, index=False)
                    print(f"Scores saved: {results_filename}")


if __name__ == '__main__':
    # Paths to training and testing files
    train_file = 'X_train.csv'

    # Load training data
    time_series = pd.read_csv(train_file, header=None).values.flatten()

    anomaly_lengths = [
        [100, 250, 500],  # Lengths for 'constant' anomalies
        [100, 250, 500],  # Lengths for 'peak' anomalies
        [100, 250, 500],  # Lengths for 'trough' anomalies
        [100, 250, 500],  # Lengths for 'reverse' anomalies
        [100, 250, 500]  # Lengths for 'noise' anomalies
    ]
    gap_between_anomalies = 2000

    test_file, labels = create_testfile_with_sequential_anomalies(
        file_path=train_file,
        anomaly_lengths=anomaly_lengths,
        gap_between_anomalies=gap_between_anomalies
    )

    parameter_grid = {
        "window_size": [100, 350],
        "k": [3, 5, 10],
        "threshold_factor": [1.5, 3, 5],
        "distance_metric": ['euclidean', 'manhattan', 'cosine']
    }

    # Output directory for results
    output_dir = "results"

    time_series = remove_trend_differencing(time_series)

    # Run parameter testing
    test_parameters(time_series, test_file, labels, parameter_grid)
