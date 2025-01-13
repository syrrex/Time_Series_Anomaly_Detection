import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from pate.PATE_metric import PATE
import pickle
from tensorflow.keras.models import load_model

# TODO: Our base model is this:
# window_size = 350
# LSTM 64 units, Dense 32 units, output_layer = Dense(window_size, activation='linear')
# Also try different parameters for this as well as the GMM.
# batch_size and epochs can be tuned as well.

file_path = "X_train.csv"
window_size = 350


def load_and_normalize_data(file_path):
    data = pd.read_csv(file_path, header=None).iloc[:, 0].values
    mean = np.mean(data)
    std_dev = np.std(data)
    normalized_data = (data - mean) / std_dev
    return normalized_data, mean, std_dev


def create_windows(data, window_size):
    return np.array([data[i:i + window_size] for i in range(len(data) - window_size + 1)])


def train_model(data, window_size, LSTM_units=64, Dense_units=32, GMM_components=2, epochs=20, batch_size=32):
    train_windows = create_windows(data, window_size).reshape(-1, window_size, 1)

    # Define LSTM Encoder
    input_layer = Input(shape=(window_size, 1))
    lstm_out = LSTM(LSTM_units, return_sequences=False)(input_layer)
    dense_out = Dense(Dense_units, activation='relu')(lstm_out)
    output_layer = Dense(window_size, activation='linear')(dense_out)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(train_windows, train_windows, epochs=epochs, batch_size=batch_size, verbose=1)

    # Extract latent features using the encoder part
    encoder = Model(inputs=input_layer, outputs=dense_out)
    latent_features = encoder.predict(train_windows)

    # Train GMM
    gmm = GaussianMixture(n_components=GMM_components, covariance_type='full', random_state=42)
    gmm.fit(latent_features)

    return encoder, gmm


if __name__ == "__main__":
    # Load and normalize data
    normalized_data, mean, std_dev = load_and_normalize_data(file_path)

    # Train the model
    encoder, gmm = train_model(normalized_data, window_size)

    # Save models
    encoder.save('encoder_model.keras')
    with open('gmm_model.pkl', 'wb') as file:
        pickle.dump(gmm, file)
