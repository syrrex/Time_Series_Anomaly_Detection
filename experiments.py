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
                                           log_transform=False,
                                           baseline=False):
    """
    Trains a KNN model with the given parameters
    """
    print(f"Training KNN model with data configuration parameters window_size={window_size}, k={k}, "
          f"distance_metric={distance_metric}, threshold_factor={threshold_factor}, log_transform={log_transform}...")

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
    if baseline:
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
    print(f"PATE score for model: {pate_metric}")
    plot_title = f"ws={window_size}_k={k}_th={threshold_factor}_dm={distance_metric}_pate={pate_metric:.4f}"
    visualize_detected_anomalies(data=pd.read_csv(path_anomaly, header=None).iloc[:, 0].values,
                                 labels=labels,
                                 scores=scores,
                                 threshold=0.75,
                                 title=plot_title)

    if smoothing:
        smoothed_scores_model = smooth_anomaly_scores(scores, sigma=smoothing_sigma)
        pate_metric_smoothed = PATE(labels, smoothed_scores_model, binary_scores=False)
        plot_title = f"ws={window_size}_k={k}_th={threshold_factor}_dm={distance_metric}_pate={pate_metric_smoothed:.4f}"
        visualize_detected_anomalies(data=pd.read_csv(path_anomaly, header=None).iloc[:, 0].values,
                                     labels=labels,
                                     scores=smoothed_scores_model,
                                     threshold=0.5,
                                     title=plot_title)
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
                                           log_transform=False,
                                           baseline=True)


# Experiment 2: We also tried grid search with log transforming the data does it work better we also smooth the scores?
def run_experiment_2():
    window_size = 250
    k = 5
    threshold_factor = 1.5
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
                                           log_transform=True)


# TODO: Experiment 3: Why does differencing not work well?
def run_experiment_3():
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
                                           differencing=True,
                                           log_transform=False,
                                           baseline=True)


# Experiment 4: What influence has the window size
def run_experiment_4():
    window_sizes = [50, 100, 400]
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

    for window_size in window_sizes:
        train_and_test_a_model_with_dataconfig(window_size=window_size,
                                               k=k,
                                               distance_metric=distance_metric,
                                               threshold_factor=threshold_factor,
                                               anomaly_length_list=anomaly_lengths,
                                               gap_between_anomalies=gap_between_anomalies,
                                               smoothing=True,
                                               differencing=False,
                                               log_transform=False,
                                               baseline=False)


# Experiment 5: What influence has threshold factor
def run_experiment_5():
    window_size = 250
    k = 2
    threshold_factors = [1, 5, 10]
    distance_metric = 'manhattan'

    anomaly_lengths = [
        [100, 250, 500],  # Lengths for 'constant' anomalies
        [100, 250, 500],  # Lengths for 'peak' anomalies
        [100, 250, 500],  # Lengths for 'trough' anomalies
        [100, 250, 500],  # Lengths for 'reverse' anomalies
        [100, 250, 500]  # Lengths for 'noise' anomalies
    ]
    gap_between_anomalies = 2000

    for threshold_factor in threshold_factors:
        train_and_test_a_model_with_dataconfig(window_size=window_size,
                                               k=k,
                                               distance_metric=distance_metric,
                                               threshold_factor=threshold_factor,
                                               anomaly_length_list=anomaly_lengths,
                                               gap_between_anomalies=gap_between_anomalies,
                                               smoothing=True,
                                               differencing=False,
                                               log_transform=False,
                                               baseline=False)


# Experiment 6: What influence has k
def run_experiment_6():
    window_size = 250
    ks = [1, 5, 10]
    threshold_factor = 2
    distance_metric = 'manhattan'

    anomaly_lengths = [
        [100, 250, 500],  # Lengths for 'constant' anomalies
        [100, 250, 500],  # Lengths for 'peak' anomalies
        [100, 250, 500],  # Lengths for 'trough' anomalies
        [100, 250, 500],  # Lengths for 'reverse' anomalies
        [100, 250, 500]  # Lengths for 'noise' anomalies
    ]
    gap_between_anomalies = 2000

    for k in ks:
        train_and_test_a_model_with_dataconfig(window_size=window_size,
                                               k=k,
                                               distance_metric=distance_metric,
                                               threshold_factor=threshold_factor,
                                               anomaly_length_list=anomaly_lengths,
                                               gap_between_anomalies=gap_between_anomalies,
                                               smoothing=True,
                                               differencing=False,
                                               log_transform=False,
                                               baseline=False)


# Experiment 7: What influence have shorter anomaly lengths
def run_experiment_7():
    window_size = 250
    k = 2
    threshold_factor = 3.5
    distance_metric = 'manhattan'

    anomaly_lengths = [
        [20, 50, 75],  # Lengths for 'constant' anomalies
        [20, 50, 75],  # Lengths for 'peak' anomalies
        [20, 50, 75],  # Lengths for 'trough' anomalies
        [20, 50, 75],  # Lengths for 'reverse' anomalies
        [20, 50, 75]  # Lengths for 'noise' anomalies
    ]
    gap_between_anomalies = 3500

    train_and_test_a_model_with_dataconfig(window_size=window_size,
                                           k=k,
                                           distance_metric=distance_metric,
                                           threshold_factor=threshold_factor,
                                           anomaly_length_list=anomaly_lengths,
                                           gap_between_anomalies=gap_between_anomalies,
                                           smoothing=True,
                                           differencing=False,
                                           log_transform=False,
                                           baseline=True)


# Experiment 8: What influence have longer anomaly lengths
def run_experiment_8():
    window_size = 250
    k = 2
    threshold_factor = 3.5
    distance_metric = 'manhattan'

    anomaly_lengths = [
        [750, 1000, 1500],  # Lengths for 'constant' anomalies
        [750, 1000, 1500],  # Lengths for 'peak' anomalies
        [750, 1000, 1500],  # Lengths for 'trough' anomalies
        [750, 1000, 1500],  # Lengths for 'reverse' anomalies
        [750, 1000, 1500]  # Lengths for 'noise' anomalies
    ]
    gap_between_anomalies = 750

    train_and_test_a_model_with_dataconfig(window_size=window_size,
                                           k=k,
                                           distance_metric=distance_metric,
                                           threshold_factor=threshold_factor,
                                           anomaly_length_list=anomaly_lengths,
                                           gap_between_anomalies=gap_between_anomalies,
                                           smoothing=True,
                                           differencing=False,
                                           log_transform=False,
                                           baseline=True)


def run_experiment_9():
    anomaly_lengths = [
        [100, 250, 500],  # Lengths for 'constant' anomalies
        [100, 250, 500],  # Lengths for 'peak' anomalies
        [100, 250, 500],  # Lengths for 'trough' anomalies
        [100, 250, 500],  # Lengths for 'reverse' anomalies
        [100, 250, 500]  # Lengths for 'noise' anomalies
    ]
    gap_between_anomalies = 2000

    path_anomaly, labels_anomaly = create_testfile_with_sequential_anomalies(
        file_path=file_path,
        anomaly_lengths=anomaly_lengths,
        gap_between_anomalies=gap_between_anomalies
    )

    data = pd.read_csv(path_anomaly, header=None).iloc[:, 0].values
    stat_data = differencing_data(data)
    visualize_test_data(stat_data, labels_anomaly)


if __name__ == '__main__':

    # Working with raw data, no preprocessing, using the best model grid search gave us
    print("Running experiment 1 with raw data, no preprocessing, using the best model grid search gave us")
    run_experiment_1()
    print("-----------------------------------------------------------------------------------------------------------")

    print("Running experiment 2 with log transforming the data, using the best model grid search gave us")
    run_experiment_2()
    print("-----------------------------------------------------------------------------------------------------------")

    print("Running experiment 3 with differencing the data")
    run_experiment_3()
    print("-----------------------------------------------------------------------------------------------------------")

    print("Running experiment 4: what influence has the window size")
    run_experiment_4()
    print("-----------------------------------------------------------------------------------------------------------")

    print("Running experiment 5: What influence has threshold factor")
    run_experiment_5()
    print("-----------------------------------------------------------------------------------------------------------")

    print("Running experiment 6: What influence has k")
    run_experiment_6()
    print("-----------------------------------------------------------------------------------------------------------")

    print("Running experiment 7: What influence have shorter anomaly lengths")
    run_experiment_7()
    print("-----------------------------------------------------------------------------------------------------------")

    print("Running experiment 8: What influence have longer anomaly lengths")
    run_experiment_8()
    print("-----------------------------------------------------------------------------------------------------------")

    # Why dont we detect extreme peaks and troughs? for this we need to inspect the data how it looks when it is stationary
    print("Bonus experiment: Why dont we detect extreme peaks and troughs when using differencing?")
    run_experiment_9()
    print("-----------------------------------------------------------------------------------------------------------")

    # TODO: Experiment 10: What happens when we change the anomaly strength
    # run_experiment_10()
