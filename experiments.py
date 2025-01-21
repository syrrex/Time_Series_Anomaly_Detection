from pate.PATE_metric import PATE
from test import *
from train import *

from create_testdata import *
from plots import *


def train_and_test_a_model_with_dataconfig(window_size,
                                           k,
                                           distance_metric,
                                           threshold_factor,
                                           anomaly_length_list,
                                           gap_between_anomalies,
                                           smoothing_sigma=75,
                                           smoothing=False,
                                           differencing=False,
                                           log_transform=False):
    """
    Trains a KNN model with the given parameters
    """
    file_path = 'X_train.csv'
    time_series = pd.read_csv(file_path, header=None).values.flatten()
    knn_model, _, scaler = train_model(
        time_series,
        window_size=window_size,
        k=k,
        distance_metric=distance_metric,
        differencing=differencing,
        log_transform=log_transform,
    )

    path_anomaly, labels = create_testfile_with_sequential_anomalies(
        file_path=file_path,
        anomaly_lengths=anomaly_length_list,
        gap_between_anomalies=gap_between_anomalies
    )

    # baseline all anomalies:
    baseline_scores = apply_baseline_model(file_path)
    pate_metric_baseline = PATE(labels, baseline_scores, binary_scores=False)
    print(f"PATE score for baseline all anomalies: {pate_metric_baseline}")

    scores = apply_anomaly_detection(
        path_anomaly,
        knn_model,
        scaler,
        window_size=window_size,
        threshold_factor=threshold_factor,
        differencing=differencing,
        log_transform=log_transform
    )

    pate_metric = PATE(labels, scores, binary_scores=False)
    visualize_detected_anomalies(path_anomaly, labels, scores, threshold=0.5)
    print(f"PATE score for model: {pate_metric}")

    if smoothing:
        smoothed_scores_model = smooth_anomaly_scores(scores, sigma=smoothing_sigma)
        pate_metric_smoothed = PATE(labels, smoothed_scores_model, binary_scores=False)
        print(f"PATE score for model with smoothing: {pate_metric_smoothed}")


# Experiment 1: Grid search gave us a model that worked with no preprocessing does it work better with smoothing?
def run_experiment_1():
    window_size = 250
    k = 2
    threshold_factor = 3.5
    distance_metric = 'manhattan'

    anomaly_lengths = [
        [100, 250, 500],  # Lengths for 'constant' anomalies
        [100, 250, 500],  # Lengths for 'peak' anomalies
        [100, 250, 500],  # Lengths for 'trough' anomalies
        [100, 250, 500],  # Lengths for 'reverse' anomalies
        [100, 250, 500]  # Lengths for 'noise' anomalies
    ]
    gap_between_anomalies = 2000

    train_and_test_a_model_with_dataconfig(window_size=window_size,
                                           k=k,
                                           distance_metric=distance_metric,
                                           threshold_factor=threshold_factor,
                                           anomaly_length_list=anomaly_lengths,
                                           gap_between_anomalies=gap_between_anomalies,
                                           smoothing=True,
                                           differencing=False,
                                           log_transform=False)

# TODO: Experiment 2: We also tried grid search with log transforming the data does it work better?
def run_experiment_2():
    pass


# TODO: Experiment 3: Why does differencing not work well?
def run_experiment_3():
    pass


#TODO: Experiment 4: Given best model, what influence has window size and k
def run_experiment_4():
    pass


# TODO: Experiment 5: Given best model, what influence has threshold factor
def run_experiment_5():
    pass

# TODO: Experiment 6: Given best model, what influence have shorter anomalie lengths
def run_experiment_6():
    pass

# TODO: Experiment 7: Given best model, what influence have longer anomalie lengths
def run_experiment_7():
    pass



if __name__ == '__main__':
    # Working with raw data, no preprocessing, using the best model grid search gave us
    run_experiment_1()

    # run_experiment_2()
    # run_experiment_3()
    # run_experiment_4()
    # run_experiment_5()
    # run_experiment_6()
    # run_experiment_7()
    # run_experiment_8()
    # run_experiment_9()
