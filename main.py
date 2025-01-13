import argparse
import pandas as pd
import numpy as np
from keras.models import load_model
import pickle
from test import *
from train import *


def main(input_file):
    output_file = "predictions_group_18.csv"

    # Load pre-trained models
    encoder = load_model('encoder_model.keras')
    with open('gmm_model.pkl', 'rb') as file:
        gmm = pickle.load(file)

    scores = apply_anomaly_detection(input_file, encoder, gmm, window_size)

    # Save scores to CSV
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(output_file, index=False, header=False)
    print(f"Anomaly scores saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly Detection Script")
    parser.add_argument("input_file", help="Path to the input CSV file")
    args = parser.parse_args()
    main(args.input_file)
