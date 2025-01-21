from pate.PATE_metric import PATE
from test import *

from create_testdata import *
from plots import *

def train_and_test_a_model_with_dataconfig(time_series,
                                           window_size,
                                           k,
                                           distance_metric,
                                           threshold_factor,
                                           anomaly_length_list,
                                           gap_between_anomalies,
                                           smoothing_sigma,
                                           smoothing=False,
                                           differencing=False,
                                           log_transform=False):
    """
    Trains a KNN model with the given parameters
    """
    file_path = 'X_train.csv'
    knn_model, _, scaler = train_model(
        time_series,
        window_size=window_size,
        k=k,
        distance_metric=distance_metric,
        differencing=differencing,
        log_transform=log_transform,
    )

    test_file, labels = create_testfile_with_sequential_anomalies(
        file_path=file_path,
        anomaly_lengths=anomaly_length_list,
        gap_between_anomalies=gap_between_anomalies
    )

    # baseline all anomalies:
    baseline_scores = apply_baseline_model(file_path)

    scores = apply_anomaly_detection(
        path_anomaly,
        knn_model1,
        scaler1,
        window_size=window_size,
        threshold_factor=threshold_factor,
        differencing=differencing,
        log_transform=log_transform
    )

    pate_metric = PATE(labels, scores, binary_scores=False)
    visualize_detected_anomalies(test_file, labels, scores, threshold=0.5)
    print(f"PATE score for model: {pate_metric}")

    if smoothing:
        smoothed_scores_model = smooth_anomaly_scores(scores, sigma=smoothing_sigma)
        pate_metric_smoothed = PATE(labels, smoothed_scores_model, binary_scores=False)
        print(f"PATE score for model with smoothing: {pate_metric_smoothed}")










def run_experiment(file_path):

if __name__ == '__main__':

