import numpy as np


def abs_fft(x):
    return np.abs(np.fft.rfft(x))


def create_fft(X):
    X_fft = np.copy(X)
    X_fft = np.apply_along_axis(abs_fft, 1, X_fft)
    return X_fft


def normalize(X, X_mean, X_std):
    # print(X.T - X_mean)
    X = X.T
    X -= X_mean
    X /= X_std
    return X.T
