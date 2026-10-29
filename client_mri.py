import numpy as np
from models.cnn_model_mri import build_cnn_mri

def train_mri_client():
    try:
        X = np.load('data/mri/X_mri.npy')
        y = np.load('data/mri/y_mri.npy')
    except:
        X = np.random.rand(5, 128, 128, 128, 1)
        y = np.array([0, 1, 1, 0, 1])

    model = build_cnn_mri()
    model.fit(X, y, epochs=1, verbose=0)
    weights = model.get_weights()
    return weights

def evaluate_mri_model(weights):
    X = np.load('data/mri/X_mri.npy')
    y = np.load('data/mri/y_mri.npy')

    model = build_cnn_mri()
    model.set_weights(weights)
    preds = (model.predict(X) > 0.5).astype(int).flatten()

    return preds, y
