import numpy as np


def min_max_scaler(X, scale_range=(0, 1)):
    scale = (scale_range[1] - scale_range[0]) / (np.max(X, axis=0) - np.min(X, axis=0))
    return scale_range[0] + (X - np.min(X, axis=0)) * scale


def train_test_split(X, y, test_size=0.25):
    np.random.seed(12345)
    X, y = np.array(X), np.array(y)
    row = X.shape[0]
    test_indices = np.random.choice(row, int(row * test_size), replace=False)
    train_indices = np.array(list(set(range(row)) - set(test_indices)))
    return X[train_indices], X[test_indices], y[train_indices].reshape(-1, 1), y[test_indices].reshape(-1, 1)

