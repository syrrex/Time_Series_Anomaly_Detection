from scipy.ndimage import gaussian_filter1d
from experiments import *
from train import *
from create_testdata import *
from pate.PATE_metric import PATE


# label every point as anomaly
def apply_baseline_model(file_path):
    time_series = pd.read_csv(file_path, header=None).iloc[:, 0].values
    scores = np.ones(len(time_series))
    return scores


def apply_anomaly_detection(file_path, model, scaler, window_size, threshold_factor, differencing=False, log_transform=False):
    # Load the test data
    test_data = pd.read_csv(file_path, header=None).iloc[:, 0].values

    if differencing:
        test_data = differencing_data(test_data)

    if log_transform:
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


def smooth_anomaly_scores(raw_scores, sigma=75):
    smoothed_data = gaussian_filter1d(raw_scores, sigma=sigma)
    return smoothed_data

if __name__ == '__main__':

    # Load the trained model and scaler
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

    knn_model = joblib.load('knn_model.joblib')
    scaler = joblib.load('scaler.joblib') if joblib.os.path.exists('scaler.joblib') else None

    window_size = 250
    k = 5
    threshold_factor = 3.5
    smoothing_sigma = 75
    distance_metric = 'manhattan'

    scores = apply_anomaly_detection(test_file, knn_model, scaler, window_size, threshold_factor, log_transform=True)
    smoothed_scores_model = smooth_anomaly_scores(scores, sigma=smoothing_sigma)
    pate_metric_smoothed = PATE(labels, smoothed_scores_model, binary_scores=False)
    print(f"PATE score for model with smoothing: {pate_metric_smoothed}")

