from train import *
from test import *
from create_testdata import *
from plots import *



if __name__ == '__main__':
    param_grid = {
        'LSTM_units': [32, 64, 128],  # Different numbers of LSTM units
        'Dense_units': [16, 32, 64],  # Different numbers of Dense units
        'GMM_components': [2, 3, 4],  # Number of GMM components
        'epochs': [10, 20, 30],  # Number of training epochs
        'batch_size': [16, 32, 64],  # Batch sizes
        'window_size': [50, 100, 350, 1000]  # Different window sizes
    }

    # TODO: Different Thresolds for anomaly detection


