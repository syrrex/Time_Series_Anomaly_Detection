from pate.PATE_metric import PATE
from scipy.ndimage import gaussian_filter1d

from test import *

from create_testdata import *
from plots import *


def smooth_large_binary_gaussian(data, sigma):
    # Apply Gaussian smoothing
    smoothed_data = gaussian_filter1d(data, sigma=sigma)
    return smoothed_data


def run_experiment_1(file_path):
    # Load the time series data
    time_series = pd.read_csv(file_path, header=None).values.flatten()

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

    window_size = 250
    k = 2
    distance_metric = 'manhattan'
    remove_trend = False
    threshold_factor = 3.5

    knn_model, _, scaler = train_model(
        time_series,
        window_size=window_size,
        k=k,
        distance_metric=distance_metric,
        remove_trend=remove_trend
    )
    anomaly_data = pd.read_csv(path_anomaly, header=None).iloc[:, 0].values

    scores = apply_anomaly_detection(path_anomaly, knn_model, scaler, window_size, threshold_factor,
                                     remove_trend=remove_trend)
    pate_metric = PATE(labels_anomaly, scores, binary_scores=False)
    visualize_detected_anomalies(anomaly_data, labels_anomaly, scores, threshold=0.9)
    print(f"PATE score for model without smoothing: {pate_metric}")

    # Smooth the scores with gaussian filter
    smoothed_scores = smooth_large_binary_gaussian(scores, sigma=75)
    pate_metric = PATE(labels_anomaly, smoothed_scores, binary_scores=False)
    print(f"PATE score for model with gaussian smoothing (sigma=75): {pate_metric}")
    visualize_detected_anomalies(anomaly_data, labels_anomaly, smoothed_scores, threshold=0.9)
    # csv_path = 'experiment_2_scores.csv'
    # pd.DataFrame(smoothed_scores).to_csv(csv_path, header=False, index=False)


def run_experiment_2(file_path):
    # Load the time series data
    time_series = pd.read_csv(file_path, header=None).values.flatten()

    anomaly_lengths = [
        [10, 50, 75],  # Lengths for 'constant' anomalies could not find smalest constant anomalie
        [10, 50, 75],  # Lengths for 'peak' anomalies
        [10, 50, 75],  # Lengths for 'trough' anomalies
        [10, 50, 75],  # Lengths for 'reverse' anomalies could not find smallest 2 reverse anomalies
        [10, 50, 75]  # Lengths for 'noise' anomalies
    ]
    gap_between_anomalies = 2000

    path_anomaly, labels_anomaly = create_testfile_with_sequential_anomalies(
        file_path=file_path,
        anomaly_lengths=anomaly_lengths,
        gap_between_anomalies=gap_between_anomalies
    )

    window_size = 250
    k = 2
    distance_metric = 'manhattan'
    remove_trend = False
    threshold_factor = 3.5

    knn_model, _, scaler = train_model(
        time_series,
        window_size=window_size,
        k=k,
        distance_metric=distance_metric,
        remove_trend=remove_trend
    )
    anomaly_data = pd.read_csv(path_anomaly, header=None).iloc[:, 0].values
    scores = apply_anomaly_detection(path_anomaly, knn_model, scaler, window_size, threshold_factor,
                                     remove_trend=remove_trend)
    pate_metric = PATE(labels_anomaly, scores, binary_scores=False)
    visualize_detected_anomalies(anomaly_data, labels_anomaly, scores, threshold=0.8)
    print(f"PATE score for model without smoothing: {pate_metric}")

    # Smooth the scores
    smoothed_scores = smooth_large_binary_gaussian(scores, sigma=75)
    pate_metric = PATE(labels_anomaly, smoothed_scores, binary_scores=False)
    visualize_detected_anomalies(anomaly_data, labels_anomaly, smoothed_scores, threshold=0.8)
    print(f"PATE score for model with smoothing: {pate_metric}")


def run_experiment_3(file_path):
    # Load the time series data
    time_series = pd.read_csv(file_path, header=None).values.flatten()

    anomaly_lengths = [
        [10, 50, 75],  # Lengths for 'constant' anomalies could not find smalest constant anomalie
        [10, 50, 75],  # Lengths for 'peak' anomalies
        [10, 50, 75],  # Lengths for 'trough' anomalies
        [10, 50, 75],  # Lengths for 'reverse' anomalies could not find smallest 2 reverse anomalies
        [10, 50, 75]  # Lengths for 'noise' anomalies
    ]
    gap_between_anomalies = 2000

    path_anomaly, labels_anomaly = create_testfile_with_sequential_anomalies(
        file_path=file_path,
        anomaly_lengths=anomaly_lengths,
        gap_between_anomalies=gap_between_anomalies
    )

    window_size = 250
    k = 2
    distance_metric = 'manhattan'
    remove_trend = False
    threshold_factor = 3.5

    knn_model, _, scaler = train_model(
        time_series,
        window_size=window_size,
        k=k,
        distance_metric=distance_metric,
        remove_trend=remove_trend
    )
    anomaly_data = pd.read_csv(path_anomaly, header=None).iloc[:, 0].values
    scores = apply_anomaly_detection(path_anomaly, knn_model, scaler, window_size, threshold_factor,
                                     remove_trend=remove_trend)
    pate_metric = PATE(labels_anomaly, scores, binary_scores=False)
    visualize_detected_anomalies(anomaly_data, labels_anomaly, scores, threshold=0.8)
    print(f"PATE score for model without smoothing: {pate_metric}")

    # Smooth the scores
    smoothed_scores = smooth_large_binary_gaussian(scores, sigma=75)
    pate_metric = PATE(labels_anomaly, smoothed_scores, binary_scores=False)
    visualize_detected_anomalies(anomaly_data, labels_anomaly, smoothed_scores, threshold=0.8)
    print(f"PATE score for model with standard smoothing: {pate_metric}")

    # Smooth the scores more aggressively
    smoothed_scores = smooth_large_binary_gaussian(scores, sigma=30)
    pate_metric = PATE(labels_anomaly, smoothed_scores, binary_scores=False)
    visualize_detected_anomalies(anomaly_data, labels_anomaly, smoothed_scores, threshold=0.8)
    print(f"PATE score for model with aggressive smoothing: {pate_metric}")


def run_experiment_4(file_path):
    # Load the time series data
    time_series = pd.read_csv(file_path, header=None).values.flatten()

    anomaly_lengths = [
        [250, 500, 1000],  # Lengths for 'constant' anomalies could not find smalest constant anomalie
        [250, 500, 1000],  # Lengths for 'peak' anomalies
        [250, 500, 1000],  # Lengths for 'trough' anomalies
        [250, 500, 1000],  # Lengths for 'reverse' anomalies could not find smallest 2 reverse anomalies
        [250, 500, 1000]  # Lengths for 'noise' anomalies
    ]
    gap_between_anomalies = 1200

    path_anomaly, labels_anomaly = create_testfile_with_sequential_anomalies(
        file_path=file_path,
        anomaly_lengths=anomaly_lengths,
        gap_between_anomalies=gap_between_anomalies
    )

    window_size = 250
    k = 2
    distance_metric = 'manhattan'
    remove_trend = False
    threshold_factor = 3.5

    knn_model, _, scaler = train_model(
        time_series,
        window_size=window_size,
        k=k,
        distance_metric=distance_metric,
        remove_trend=remove_trend
    )
    anomaly_data = pd.read_csv(path_anomaly, header=None).iloc[:, 0].values
    scores = apply_anomaly_detection(path_anomaly, knn_model, scaler, window_size, threshold_factor,
                                     remove_trend=remove_trend)
    pate_metric = PATE(labels_anomaly, scores, binary_scores=False)
    visualize_detected_anomalies(anomaly_data, labels_anomaly, scores, threshold=0.8)
    print(f"PATE score for model without smoothing: {pate_metric}")

    # Smooth the scores
    smoothed_scores = smooth_large_binary_gaussian(scores, sigma=75)
    pate_metric = PATE(labels_anomaly, smoothed_scores, binary_scores=False)
    visualize_detected_anomalies(anomaly_data, labels_anomaly, smoothed_scores, threshold=0.8)
    print(f"PATE score for model with standard smoothing: {pate_metric}")

    # Smooth the scores more aggressively
    smoothed_scores = smooth_large_binary_gaussian(scores, sigma=100)
    pate_metric = PATE(labels_anomaly, smoothed_scores, binary_scores=False)
    visualize_detected_anomalies(anomaly_data, labels_anomaly, smoothed_scores, threshold=0.8)
    print(f"PATE score for model with less smoothing: {pate_metric}")


def run_experiment_5(file_path):
    # Load the time series data
    time_series = pd.read_csv(file_path, header=None).values.flatten()

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

    window_size = 250
    ks = [2, 5, 10, 20]
    distance_metric = 'manhattan'
    remove_trend = False
    threshold_factor = 3.5

    for k in ks:
        knn_model, _, scaler = train_model(
            time_series,
            window_size=window_size,
            k=k,
            distance_metric=distance_metric,
            remove_trend=remove_trend
        )
        anomaly_data = pd.read_csv(path_anomaly, header=None).iloc[:, 0].values

        scores = apply_anomaly_detection(path_anomaly, knn_model, scaler, window_size, threshold_factor,
                                         remove_trend=remove_trend)

        # Smooth the scores with gaussian filter
        smoothed_scores = smooth_large_binary_gaussian(scores, sigma=75)
        pate_metric = PATE(labels_anomaly, smoothed_scores, binary_scores=False)
        print(f"PATE score for model with gaussian smoothing and k={k}: {pate_metric}")
        visualize_detected_anomalies(anomaly_data, labels_anomaly, smoothed_scores, threshold=0.9)
    # csv_path = 'experiment_2_scores.csv'
    # pd.DataFrame(smoothed_scores).to_csv(csv_path, header=False, index=False)


def run_experiment_6(file_path):
    # Load the time series data
    time_series = pd.read_csv(file_path, header=None).values.flatten()

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

    window_sizes = [50, 100, 250, 400]
    k = 2
    distance_metric = 'manhattan'
    remove_trend = False
    threshold_factor = 3.5

    for window_size in window_sizes:
        knn_model, _, scaler = train_model(
            time_series,
            window_size=window_size,
            k=k,
            distance_metric=distance_metric,
            remove_trend=remove_trend
        )
        anomaly_data = pd.read_csv(path_anomaly, header=None).iloc[:, 0].values

        scores = apply_anomaly_detection(path_anomaly, knn_model, scaler, window_size, threshold_factor,
                                         remove_trend=remove_trend)

        # Smooth the scores with gaussian filter
        smoothed_scores = smooth_large_binary_gaussian(scores, sigma=75)
        pate_metric = PATE(labels_anomaly, smoothed_scores, binary_scores=False)
        print(f"PATE score for model with gaussian smoothing and window_size={window_size}: {pate_metric}")
        visualize_detected_anomalies(anomaly_data, labels_anomaly, smoothed_scores, threshold=0.5)


def run_experiment_7(file_path):
    time_series = pd.read_csv(file_path, header=None).values.flatten()

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

    window_size = 250
    k = 2
    distance_metric = 'manhattan'
    remove_trend = True
    threshold_factor = 3.5

    knn_model, _, scaler = train_model(
        time_series,
        window_size=window_size,
        k=k,
        distance_metric=distance_metric,
        remove_trend=remove_trend
    )
    anomaly_data = pd.read_csv(path_anomaly, header=None).iloc[:, 0].values

    scores = apply_anomaly_detection(path_anomaly, knn_model, scaler, window_size, threshold_factor,
                                     remove_trend=remove_trend)
    pate_metric = PATE(labels_anomaly, scores, binary_scores=False)
    visualize_detected_anomalies(anomaly_data, labels_anomaly, scores, threshold=0.9)
    print(f"PATE score for model without smoothing: {pate_metric}")

    # Smooth the scores with gaussian filter
    smoothed_scores = smooth_large_binary_gaussian(scores, sigma=75)
    pate_metric = PATE(labels_anomaly, smoothed_scores, binary_scores=False)
    print(f"PATE score for model with gaussian smoothing (sigma=75): {pate_metric}")
    visualize_detected_anomalies(anomaly_data, labels_anomaly, smoothed_scores, threshold=0.9)


def run_experiment_8(file_path):
    # time_series = pd.read_csv(file_path, header=None).values.flatten()

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
    stat_data = make_stationary_differencing(data)
    visualize_test_data(stat_data, labels_anomaly)


def run_experiment_9(file_path):
    # time_series = pd.read_csv(file_path, header=None).values.flatten()

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

    time_series = pd.read_csv(path_anomaly, header=None).iloc[:, 0].values
    stat_data = log_transform_data(time_series)
    visualize_test_data(stat_data, labels_anomaly)

    window_size = 250
    k = 2
    distance_metric = 'manhattan'
    diifferencing = False
    log_transform = True
    threshold_factor = 3.5

    knn_model, _, scaler = train_model(
        time_series,
        window_size=window_size,
        k=k,
        distance_metric=distance_metric,
        remove_trend=diifferencing,
        log_transform=log_transform
    )
    anomaly_data = pd.read_csv(path_anomaly, header=None).iloc[:, 0].values

    scores = apply_anomaly_detection(path_anomaly, knn_model, scaler, window_size, threshold_factor,
                                     remove_trend=diifferencing, log_transform=log_transform)
    pate_metric = PATE(labels_anomaly, scores, binary_scores=False)
    visualize_detected_anomalies(anomaly_data, labels_anomaly, scores, threshold=0.9)
    print(f"PATE score for model without smoothing: {pate_metric}")

    # Smooth the scores with gaussian filter
    smoothed_scores = smooth_large_binary_gaussian(scores, sigma=75)
    pate_metric = PATE(labels_anomaly, smoothed_scores, binary_scores=False)
    print(f"PATE score for model with gaussian smoothing (sigma=75): {pate_metric}")
    visualize_detected_anomalies(anomaly_data, labels_anomaly, smoothed_scores, threshold=0.9)




if __name__ == '__main__':
    # Path to your CSV file
    file_path = 'X_train.csv'

    # now that we found a suitable model we can make different experiments:
    # Experiment 1: What happens if we smooth the scores?
    # run_experiment_1(file_path)

    # Experiment 2: What happens if we make anomaly sequences very small?
    # run_experiment_2(file_path)

    # TODO: if we know we test for smaller anomaly sequences, we could increase our smoothing
    # Experiment 3:
    # run_experiment_3(file_path)

    # TODO: What happens if we make anomaly sequences very large and is less smoothing better for large anomaly detection?
    # Experiment 4:
    # run_experiment_4(file_path)

    # TODO: what influence has k on it
    # Experiment 5:
    # run_experiment_5(file_path)

    # TODO: what influence has window_size on it
    # Experiment 6:
    # run_experiment_6(file_path)

    # TODO: what happens if we difference the time series and make it stationary?
    # Experiment 7: also experimenting with grid search to find a valid combination of parameters did not lead to valid results
    # Problem are smoothing makes results worse, also extreme peaks and troughs are not detected
    # run_experiment_7(file_path)

    # TODO: Why dont we detect extreme peaks and troughs? for this we need to inspect the data how it looks when it is stationary
    # Experiment 8: after differencing the time series, we can see that sequences of extrem peaks and troughs are just one single anomaly point that can not be detected
    # run_experiment_8(file_path)

    # TODO: Can we use something different than differencing to make the time series stationary? i.e. log-transform
    # Experiment 9:
    run_experiment_9(file_path)

    # TODO: What happens if we use different anomalies, smaller peaks
