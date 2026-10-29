import numpy as np
from models.rnn_model_eeg import build_rnn_eeg

def train_eeg_client():
    try:
        X = np.load('data/eeg/X_eeg.npy')
        y = np.load('data/eeg/y_eeg.npy')
    except:
        X = np.random.rand(5, 100, 64)
        y = np.array([0, 1, 1, 0, 1])

    model = build_rnn_eeg()
    model.fit(X, y, epochs=1, verbose=0)
    weights = model.get_weights()
    return weights

def evaluate_eeg_model(weights):
    X = np.load('data/eeg/X_eeg.npy')
    y = np.load('data/eeg/y_eeg.npy')

    model = build_rnn_eeg()
    model.set_weights(weights)
    preds = (model.predict(X) > 0.5).astype(int).flatten()

    return preds, y
