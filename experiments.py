from pate.PATE_metric import PATE
from test import *

from create_testdata import *
from plots import *

if __name__ == '__main__':
    # Path to your CSV file
    file_path = 'X_train.csv'

    # Load the time series data
    time_series = pd.read_csv(file_path, header=None).values.flatten()

    # 1. Experiment: Train the two best looking models that we found using the grid search:
    # Parameters for the most stable model found in with grid search
    window_sizes = [100, 350]
    k = 3
    distance_metric = ['euclidean', 'manhattan']
    threshold = 1.5

    # Train model1
    knn_model1, _, scaler1 = train_model(
        time_series,
        window_size=window_sizes[0],
        k=k,
        distance_metric=distance_metric[0]
    )

    # Train model2
    knn_model2, _, scaler2 = train_model(
        time_series,
        window_size=window_sizes[1],
        k=k,
        distance_metric=distance_metric[1]
    )
    print("Model trained successfully!")

    # # test the two best looking models with different anomaly lengths
    anomaly_lengths_list = [100, 350, 500]
    for anomaly_length in anomaly_lengths_list:
        print(f"Processing dataset with anomaly length: {anomaly_length}")
        anomaly_lengths = [anomaly_length] * 5

        path_anomaly, labels_anomaly = create_testfile_with_one_anomaly_each(
            file_path, anomaly_lengths
        )

        # baseline all anomalies:
        baseline_scores = apply_baseline_model(file_path)

        scores_model1 = apply_anomaly_detection(
            path_anomaly,
            knn_model1,
            scaler1,
            window_size=window_sizes[0],
            threshold_factor=threshold,
        )
        scores_model2 = apply_anomaly_detection(
            path_anomaly,
            knn_model2,
            scaler2,
            window_size=window_sizes[1],
            threshold_factor=threshold,
        )

        pate_metric_1 = PATE(labels_anomaly, scores_model1, binary_scores=False)
        pate_metric_2 = PATE(labels_anomaly, scores_model2, binary_scores=False)

        data = pd.read_csv(path_anomaly, header=None).iloc[:, 0].values
        visualize_detected_anomalies(data, labels_anomaly, scores_model1, threshold=0.5)
        visualize_detected_anomalies(data, labels_anomaly, scores_model2, threshold=0.5)

        baseline_scores = PATE(labels_anomaly, baseline_scores, binary_scores=False)
        print(f"PATE score for baseline model: {baseline_scores}")

        print(f"PATE score for model1 without smoothing: {pate_metric_1}")
        print(f"PATE score for model2 without smoothing: {pate_metric_2}")

        smoothed_scores_1 = smooth_anomaly_scores(scores_model1, window_size=window_sizes[0])
        smoothed_scores_2 = smooth_anomaly_scores(scores_model2, window_size=window_sizes[1])

        pate_metric_1_smoothed = PATE(labels_anomaly, smoothed_scores_1, binary_scores=False)
        pate_metric_2_smoothed = PATE(labels_anomaly, smoothed_scores_2, binary_scores=False)
        print(f"PATE score for model1 with smoothing: {pate_metric_1_smoothed}")
        print(f"PATE score for model2 with smoothing: {pate_metric_2_smoothed}")

    # Experiment: Different k: slightly worse results with higher k but not that significant
    k_list = [3, 5, 7, 9]
    for k in k_list:
        print(f"Processing dataset with k: {k}")
        anomaly_lengths = [250, 250, 250, 250, 250]
        path_anomaly, labels_anomaly = create_testfile_with_one_anomaly_each(
            file_path, anomaly_lengths
        )
        anomaly_data = pd.read_csv(path_anomaly, header=None).iloc[:, 0].values
        if k == 1:
            baseline_scores = apply_baseline_model(file_path)
            pate_baseline = PATE(labels_anomaly, baseline_scores, binary_scores=False)
            print(f"PATE score for baseline model: {pate_baseline}")

        knn_model, _, scaler = train_model(
            time_series,
            window_size=window_sizes[1],
            k=k,
            distance_metric=distance_metric[1]
        )

        scores = apply_anomaly_detection(
            path_anomaly,
            knn_model,
            scaler,
            window_size=window_sizes[1],
            threshold_factor=threshold,
        )
        smoothed_scores = smooth_anomaly_scores(scores, window_size=window_sizes[1])

        pate_metric = PATE(labels_anomaly, smoothed_scores, binary_scores=False)
        visualize_detected_anomalies(anomaly_data, labels_anomaly, smoothed_scores, threshold=0.5)
        print(f"PATE score for model with k={k}: {pate_metric}")

    # Experiment: Different Threshold
    thres_list = [2, 2.4, 2.5, 2.6, 3]
    for threshold in thres_list:
        print(f"Processing dataset with threshold: {threshold}")
        anomaly_lengths = [250, 250, 250, 250, 250]
        path_anomaly, labels_anomaly = create_testfile_with_one_anomaly_each(
            file_path, anomaly_lengths
        )
        anomaly_data = pd.read_csv(path_anomaly, header=None).iloc[:, 0].values
        if threshold == 2:
            baseline_scores = apply_baseline_model(file_path)
            pate_baseline = PATE(labels_anomaly, baseline_scores, binary_scores=False)
            print(f"PATE score for baseline model: {pate_baseline}")

        knn_model, _, scaler = train_model(
            time_series,
            window_size=window_sizes[1],
            k=k,
            distance_metric=distance_metric[1]
        )

        scores = apply_anomaly_detection(
            path_anomaly,
            knn_model,
            scaler,
            window_size=window_sizes[1],
            threshold_factor=threshold,
        )
        smoothed_scores = smooth_anomaly_scores(scores, window_size=window_sizes[1])

        pate_metric = PATE(labels_anomaly, smoothed_scores, binary_scores=False)
        visualize_detected_anomalies(anomaly_data, labels_anomaly, smoothed_scores, threshold=0.5)
        print(f"PATE score for model with threshold={threshold}: {pate_metric}")
