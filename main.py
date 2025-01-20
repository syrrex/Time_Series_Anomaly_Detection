import argparse

from test import *
from train import *


def main(input_file):
    output_file = "predictions_group_18.csv"

    # Load pre-trained models
    knn_model = joblib.load('knn_model.joblib')
    scaler = joblib.load('scaler.joblib') if joblib.os.path.exists('scaler.joblib') else None

    window_size = 350
    threshold_factor = 2.5
    scores = apply_anomaly_detection(input_file, knn_model, scaler, window_size, threshold_factor)
    smoothed_scores = smooth_anomaly_scores(scores, window_size)
    # Save scores to CSV
    scores_df = pd.DataFrame(smoothed_scores)
    scores_df.to_csv(output_file, index=False, header=False)
    print(f"Anomaly scores saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly Detection Script")
    parser.add_argument("input_file", help="Path to the input CSV file")
    args = parser.parse_args()
    main(args.input_file)
