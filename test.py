from pate.PATE_metric import PATE
from scipy.ndimage import gaussian_filter1d

from train import *
from create_testdata import *
from plots import *
from create_testdata import *


def apply_anomaly_detection(file_path, model, scaler, window_size, threshold_factor):
    # Load the test data
    test_data = pd.read_csv(file_path, header=None).iloc[:, 0].values

    # Normalize the data if a scaler is provided
    if scaler:
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
    # Load the model and scaler
    knn_model = joblib.load('knn_model.joblib')
    scaler = joblib.load('scaler.joblib') if os.path.exists('scaler.joblib') else None

    # Generate datasets
    anomaly_lengths = [350, 200, 200, 1000, 200, 5000]
    gap_between_anomalies = 5000

    filepath, labels = create_testfile_with_sequential_anomalies(
        file_path,
        anomaly_lengths=anomaly_lengths,
        gap_between_anomalies=gap_between_anomalies
    )
    data = pd.read_csv(filepath, header=None).iloc[:, 0].values
    visualize_test_data(data, labels)

    scores = apply_anomaly_detection(
        filepath,
        knn_model,
        scaler,
        window_size=350,
        threshold_factor=3.5
    )

    smoothed_scores = smooth_anomaly_scores(scores, window_size=350)

    base_dir = os.path.dirname(filepath)
    output_path = os.path.join(base_dir, "anomaly_scores.csv")
    df = pd.DataFrame(smoothed_scores, columns=["Score"])
    df.to_csv(output_path, index=False, header=False)

    visualize_detected_anomalies(data, labels, smoothed_scores, threshold=0.8)

    # Compute PATE metric
    print("compute metric")
    pate_metric = PATE(labels, smoothed_scores, binary_scores=False)
    print(f"Dataset: PATE Score: {pate_metric:.4f}")
