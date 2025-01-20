from train import *
from test import *
from create_testdata import *
from plots import *


if __name__ == '__main__':
    # Path to your CSV file
    file_path = 'X_train.csv'

    # Load the time series data
    time_series = pd.read_csv(file_path, header=None).values.flatten()

    # baseline all anomalies:
    baseline_scores = apply_baseline_model(file_path)

    # Train the two best looking models:
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

    # test the two best looking models with different anomaly lengths
    anomaly_lengths_list = [100, 350, 500]
    for anomaly_length in anomaly_lengths_list:
        print(f"Processing dataset with anomaly length: {anomaly_length}")
        anomaly_lengths = [anomaly_length] * 5

        path_anomaly, labels_anomaly = create_testfile_with_one_anomaly_each(
            file_path, anomaly_lengths
        )

        scores_model1 = apply_anomaly_detection(
            path_anomaly,
            knn_model1,
            scaler1,
            window_size=window_sizes[0],
            threshold_factor=threshold
        )
        scores_model2 = apply_anomaly_detection(
            path_anomaly,
            knn_model2,
            scaler2,
            window_size=window_sizes[1],
            threshold_factor=threshold
        )

        pate_metric_1 = PATE(labels_anomaly, scores_model1, binary_scores=False)
        pate_metric_2 = PATE(labels_anomaly, scores_model2, binary_scores=False)

        data = pd.read_csv(file_path, header=None).iloc[:, 0].values
        visualize_detected_anomalies(data, labels_anomaly, scores_model1, threshold=0.5)
        visualize_detected_anomalies(data, labels_anomaly, scores_model2, threshold=0.5)

        print(f"PATE score for model1 without smoothing: {pate_metric_1}")
        print(f"PATE score for model2 without smoothing: {pate_metric_2}")

        smoothed_scores_1 = smooth_anomaly_scores(scores_model1, window_size=window_sizes[0])
        smoothed_scores_2 = smooth_anomaly_scores(scores_model2, window_size=window_sizes[1])

        pate_metric_1_smoothed = PATE(labels_anomaly, smoothed_scores_1, binary_scores=False)
        pate_metric_2_smoothed = PATE(labels_anomaly, smoothed_scores_2, binary_scores=False)
        print(f"PATE score for model1 with smoothing: {pate_metric_1_smoothed}")
        print(f"PATE score for model2 with smoothing: {pate_metric_2_smoothed}")




