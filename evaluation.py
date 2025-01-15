import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam
import itertools
import pickle

# For anomaly detection metrics
from pate.PATE_metric import PATE
from tensorflow.keras.models import load_model


# 1) LOAD AND NORMALIZE DATA
# ----------------------------------------------------
def load_and_normalize_data(file_path):
    """
    Loads a CSV with a single column (value) and returns:
      - normalized_data
      - mean
      - std_dev
    """
    data = pd.read_csv(file_path, header=None).iloc[:, 0].values
    mean = np.mean(data)
    std_dev = np.std(data)
    normalized_data = (data - mean) / std_dev
    return normalized_data, mean, std_dev


def load_test_data_with_labels(file_path, mean, std_dev):
    """
    Loads a CSV assumed to have columns ['value', 'anomaly_label'] and returns:
      - normalized_values
      - labels (as integer array)
    Normalization uses the provided mean and std_dev from training.
    """
    df = pd.read_csv(file_path)
    # e.g., columns: value, anomaly_label
    values = df['value'].values
    labels = df['anomaly_label'].values.astype(int)
    
    normalized_values = (values - mean) / std_dev
    return normalized_values, labels


def create_windows(data, window_size):
    """
    Reshapes the time-series into overlapping windows of shape
    (num_windows, window_size).
    """
    return np.array([data[i:i + window_size] 
                     for i in range(len(data) - window_size + 1)])


# 2) BUILD & TRAIN MODEL
# ----------------------------------------------------
def build_autoencoder(window_size, LSTM_units=64, Dense_units=32):
    """
    Builds a simple autoencoder with:
      - LSTM (LSTM_units)
      - Dense (Dense_units)
      - Output shape = window_size
    Returns compiled model.
    """
    input_layer = Input(shape=(window_size, 1))
    lstm_out = LSTM(LSTM_units, return_sequences=False)(input_layer)
    dense_out = Dense(Dense_units, activation='relu')(lstm_out)
    output_layer = Dense(window_size, activation='linear')(dense_out)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(), loss='mse')
    return model


def train_model(data, window_size, 
                LSTM_units=64, 
                Dense_units=32, 
                GMM_components=2, 
                epochs=20, 
                batch_size=32):
    """
    Trains an LSTM autoencoder on 'data' then fits a GMM on the latent space.
    Returns (autoencoder, encoder, gmm).
    """
    # Prepare training windows
    train_windows = create_windows(data, window_size)
    train_windows = train_windows.reshape(-1, window_size, 1)

    # Build the autoencoder
    autoencoder = build_autoencoder(window_size, LSTM_units, Dense_units)

    # Train
    autoencoder.fit(train_windows, 
                    train_windows, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    verbose=1)  # verbose=0 for cleaner output

    # Build an encoder model (same inputs, up to dense_out)
    encoder = Model(inputs=autoencoder.input, 
                    outputs=autoencoder.layers[-2].output)

    # Extract latent features
    latent_features = encoder.predict(train_windows)

    # Fit GMM on latent features
    gmm = GaussianMixture(n_components=GMM_components,
                          covariance_type='full',
                          random_state=42)
    gmm.fit(latent_features)

    return autoencoder, encoder, gmm


# 3) EVALUATION: ANOMALY DETECTION
# ----------------------------------------------------
def compute_anomaly_scores(encoder, gmm, data, window_size):
    """
    Given an encoder (Model) and a GMM, computes the anomaly score
    for each window in 'data' (which is assumed to be normalized).
    
    Returns an array of scores, one per window (higher => more anomalous).
    """
    windows = create_windows(data, window_size).reshape(-1, window_size, 1)
    latent_features = encoder.predict(windows)  # shape: (num_windows, Dense_units)
    
    # GMM log-likelihood for each window's latent vector
    log_likelihood = gmm.score_samples(latent_features)
    
    # Typically, "higher negative log-likelihood => more anomalous"
    # So we define anomaly_score = -log_likelihood
    anomaly_scores = -log_likelihood
    
    return anomaly_scores


def align_scores_with_labels(scores, labels, window_size):
    """
    Because we create windows from time-series (length N -> N - window_size + 1 windows),
    we need to align each window's anomaly score with the appropriate label(s).

    Simplest approach: 
      - Associate each window's score with its last index or its center index.
      - Or replicate the score across the entire window range in a naive approach.

    Below is a minimal approach: 
      - We'll align each window's score with the *last time-step* in that window.
      - That means scores[i] goes with labels at index i+window_size-1.
    """
    num_windows = len(scores)
    # Create an array for time steps
    full_length = len(labels)
    aligned_scores = np.zeros(full_length)  # initialize
    aligned_scores[:] = np.nan  # for places that can't be scored
    
    for i in range(num_windows):
        last_index = i + window_size - 1
        aligned_scores[last_index] = scores[i]
    
    # Return only where we have valid (non-NaN) scores and corresponding labels
    valid_mask = ~np.isnan(aligned_scores)
    valid_scores = aligned_scores[valid_mask]
    valid_labels = labels[valid_mask]
    
    return valid_scores, valid_labels


def evaluate_anomaly_performance(scores, labels, method='max_f1'):
    """
    Uses the PATE metric for anomaly detection. 
    'method' can be 'max_f1', 'max_precision', 'max_recall', etc.
    Returns a dictionary with F1, precision, recall, threshold used, etc.
    """
    # Initialize PATE
    pate = PATE(labels=labels, scores=scores, threshold=method)
    result = pate.get_metric()  # returns a dict with e.g. 'precision', 'recall', 'f1'
    return result


# 4) SENSITIVITY STUDY
# ----------------------------------------------------
def sensitivity_study(train_path, test_path, window_size, param_grid):
    """
    Performs a simple sensitivity study by iterating over all hyperparameter 
    combos in param_grid, training on (train_path) and evaluating on (test_path).
    
    param_grid is a dict with lists, e.g.:
      {
         'LSTM_units': [32, 64],
         'Dense_units': [16, 32],
         'GMM_components': [2, 3],
         'epochs': [5, 10],
         'batch_size': [16, 32]
      }
    """
    # 4.1) LOAD TRAIN AND TEST DATA
    # -----------------------------
    # Load + normalize training data
    train_data, mean, std_dev = load_and_normalize_data(train_path)
    
    # Load + normalize test data (keeping ground truth anomaly labels)
    test_values, test_labels = load_test_data_with_labels(test_path, mean, std_dev)
    
    # 4.2) PREPARE A RESULTS LIST
    results = []
    
    # 4.3) ITERATE OVER HYPERPARAMETER COMBINATIONS
    # ---------------------------------------------
    for (LSTM_units,
         Dense_units,
         GMM_components,
         epochs,
         batch_size) in itertools.product(
            param_grid['LSTM_units'],
            param_grid['Dense_units'],
            param_grid['GMM_components'],
            param_grid['epochs'],
            param_grid['batch_size']
         ):
        # Train
        autoencoder, encoder, gmm = train_model(
            train_data,
            window_size,
            LSTM_units=LSTM_units,
            Dense_units=Dense_units,
            GMM_components=GMM_components,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Compute anomaly scores on test
        scores = compute_anomaly_scores(encoder, gmm, test_values, window_size)
        
        # Align scores with labels
        aligned_scores, aligned_labels = align_scores_with_labels(scores, test_labels, window_size)
        
        # Evaluate using PATE
        performance = evaluate_anomaly_performance(aligned_scores, aligned_labels, method='max_f1')
        
        # Collect results
        single_result = {
            'LSTM_units'    : LSTM_units,
            'Dense_units'   : Dense_units,
            'GMM_components': GMM_components,
            'epochs'        : epochs,
            'batch_size'    : batch_size,
            'precision'     : performance.get('precision', np.nan),
            'recall'        : performance.get('recall', np.nan),
            'f1'            : performance.get('f1', np.nan),
            'threshold'     : performance.get('threshold', np.nan)
        }
        results.append(single_result)
        
        print(f"Tested: LSTM={LSTM_units}, Dense={Dense_units}, "
              f"GMM={GMM_components}, E={epochs}, B={batch_size} | "
              f"F1={single_result['f1']:.4f} (threshold={single_result['threshold']:.4f})")
    
    # 4.4) CONVERT RESULTS TO A DATAFRAME
    df_results = pd.DataFrame(results)
    return df_results


# 5) MAIN SCRIPT
# ----------------------------------------------------
if __name__ == "__main__":
    # Example hyperparameter grid
    param_grid = {
        'LSTM_units': [32, 64],
        'Dense_units': [16, 32],
        'GMM_components': [2, 3],
        'epochs': [10],
        'batch_size': [16, 32]
    }
    
    train_path = "X_train.csv"
    test_path  = "X_test.csv"
    window_size = 350
    
    # Run the sensitivity study
    results_df = sensitivity_study(train_path, test_path, window_size, param_grid)
    
    print("\n=== SENSITIVITY STUDY RESULTS ===")
    print(results_df)
    
    # Sort by best F1 (descending)
    sorted_df = results_df.sort_values(by="f1", ascending=False)
    print("\nTop configurations by F1 score:")
    print(sorted_df.head())

    # Optionally save to CSV
    sorted_df.to_csv("sensitivity_results.csv", index=False)
    print("\nSaved sensitivity study results to sensitivity_results.csv")
