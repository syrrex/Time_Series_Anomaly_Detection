from train import *
from create_testdata import *
from plots import *

def apply_anomaly_detection(file_path, encoder, gmm, window_size):

    # Load and normalize data
    normalized_data, mean, std_dev = load_and_normalize_data(file_path)

    # Create sliding windows
    test_windows = create_windows(normalized_data, window_size).reshape(-1, window_size, 1)

    # Extract latent features using encoder
    test_latent_features = encoder.predict(test_windows)

    # Compute log-likelihoods using GMM
    log_likelihoods = gmm.score_samples(test_latent_features)

    # Normalize log-likelihoods to scores (between 0 and 1)
    min_log_likelihood = np.min(log_likelihoods)
    max_log_likelihood = np.max(log_likelihoods)
    scores = 1 - (log_likelihoods - min_log_likelihood) / (max_log_likelihood - min_log_likelihood)

    # Handle the first window_size - 1 points using smaller windows
    initial_scores = []
    for size in range(1, window_size):
        smaller_window = normalized_data[:size].reshape(1, size, 1)
        smaller_features = encoder.predict(smaller_window)
        log_likelihood = gmm.score_samples(smaller_features)
        normalized_score = 1 - (log_likelihood[0] - min_log_likelihood) / (max_log_likelihood - min_log_likelihood)
        initial_scores.append(normalized_score)

    full_scores = np.concatenate([initial_scores, scores])
    # TODO: Apply threshold to detect anomalies test different threshold on different datasets

    return full_scores


if __name__ == '__main__':
    # Load models
    encoder = load_model('encoder_model.keras')
    with open('gmm_model.pkl', 'rb') as file:
        gmm = pickle.load(file)

    # Generate datasets
    filepath, labels = create_testdata(file_path)
    data = pd.read_csv(filepath, header=None).iloc[:, 0].values
    visualize_test_data(data, labels)

    # Apply anomaly detection
    scores = apply_anomaly_detection(filepath, encoder, gmm, window_size)

    base_dir = os.path.dirname(filepath)
    output_path = os.path.join(base_dir, "anomaly_scores.csv")
    df = pd.DataFrame(scores, columns=["Score"])
    df.to_csv(output_path, index=False, header=False)

    visualize_detected_anomalies(data, labels, scores, threshold=0.5)

    # Compute PATE metric
    print("compute metric")
    pate_metric = PATE(labels, scores, binary_scores=False)
    print(f"Dataset: PATE Score: {pate_metric:.4f}")

