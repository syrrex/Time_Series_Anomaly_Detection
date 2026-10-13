from pate.PATE_metric import PATE

from train import *
from create_testdata import *


# label every point as anomaly
def apply_baseline_model(file_path):
    time_series = pd.read_csv(file_path, header=None).iloc[:, 0].values
    scores = np.ones(len(time_series))
    return scores


def apply_anomaly_detection(file_path, model, scaler, window_size, threshold_factor, remove_trend=True, log_transform=False):
    # Load the test data
    test_data = pd.read_csv(file_path, header=None).iloc[:, 0].values

    if remove_trend:
        test_data = make_stationary_differencing(test_data)
    elif log_transform:
        test_data = log_transform_data(test_data)

    test_data = scaler.transform(test_data.reshape(-1, 1)).flatten()

    # Create sliding windows for the test data
    windows = []
    for i in range(len(test_data) - window_size + 1):
        windows.append(test_data[i:i + window_size])
    windows = np.array(windows)

    # Compute distances to nearest neighbors using the model
    distances, _ = model.kneighbors(windows)
    mean_distances = np.mean(distances, axis=1)

    # Determine the anomaly threshold
    threshold = threshold_factor * np.median(mean_distances)

    # Generate anomaly scores
    scores = np.zeros(len(test_data))
    for i, dist in enumerate(mean_distances):
        if dist > threshold:
            scores[i:i + window_size] = 1

    # Make sure the scores array matches the length of the test data
    scores = scores[:len(test_data)]

    return scores


def smooth_anomaly_scores(raw_scores, window_size):
    smoothed_scores = np.zeros_like(raw_scores, dtype=float)

    for i in range(len(raw_scores)):
        if i < window_size:
            # Beginning: Average over available scores
            smoothed_scores[i] = np.mean(raw_scores[:i + 1])
        elif i > len(raw_scores) - window_size:
            # End: Average over available scores
            smoothed_scores[i] = np.mean(raw_scores[i:])
        else:
            # Middle: Average within the window
            smoothed_scores[i] = np.mean(raw_scores[i - window_size // 2: i + window_size // 2 + 1])

    return smoothed_scores


if __name__ == '__main__':
    file_path = 'X_train.csv'
    time_series = pd.read_csv(file_path, header=None).values.flatten()
    # Load the trained model and scaler
    knn_model = joblib.load('knn_model.joblib')
    scaler = joblib.load('scaler.joblib') if os.path.exists('scaler.joblib') else None
    threshold = 3.5
    k = 3
    anomaly_lengths = [350, 350, 350, 350, 350]
    path_anomaly, labels_anomaly = create_testfile_with_one_anomaly_each(
        file_path, anomaly_lengths
    )
    anomaly_data = pd.read_csv(path_anomaly, header=None).iloc[:, 0].values
    scores = apply_anomaly_detection(
        path_anomaly,
        knn_model,
        scaler,
        window_size=350,
        threshold_factor=threshold,
        remove_trend=False
    )
    smoothed_scores = smooth_anomaly_scores(scores, window_size=350)

    pate_metric = PATE(labels_anomaly, smoothed_scores, binary_scores=False)
    visualize_detected_anomalies(anomaly_data, labels_anomaly, smoothed_scores, threshold=0.5)
    print(f"PATE score for model with k={k}: {pate_metric}")
