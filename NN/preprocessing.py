from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import scipy
import scipy.signal


class ID(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return X

    def transform(self, X, y=None):
        return X


class FrequencyRanges(BaseEstimator, TransformerMixin):
    def __init__(self, frequencies=None):
        if frequencies is None:
            self.frequencies = [(1, 4), (4, 8), (8, 12), (12, 30), (30, 50)]

    def fit(self, X, y=None):
        return self

    def GetButterSignal(self, fs: int, low: float, high: float, signal):
        """

        Parameters
        ----------
        fs : int
            frequency
        low : int
            low bracket
        high : int
            high bracket
        signal : ndarray
            1d numpy array to calculate on
        Returns
        -------
        out : ndarray
        """
        order = 4  # approximation
        nyq = 0.5 * fs  # discretion's frequency
        low = low / nyq  # lower bracket
        high = high / nyq  # top bracket
        b, a = scipy.signal.butter(order, [low, high], 'bandpass', analog=False)

        y = scipy.signal.filtfilt(b, a, signal)
        return y

    def transform(self, X, y=None, **fit_params):
        frequency_ranges_count = len(self.frequencies)
        columns_count = X.shape[-1]

        X_new = np.zeros((X.shape[0], X.shape[1], X.shape[2] * frequency_ranges_count))
        for i in range(len(self.frequencies)):
            for j in range(columns_count):
                X_new[:, :, j * frequency_ranges_count + i] = \
                    self.GetButterSignal(X.shape[1], self.frequencies[i][0],
                                         self.frequencies[i][1], X[:,:,j])

        return X_new

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X, y=None, **fit_params)


class Normalize(TransformerMixin):
    @staticmethod
    def normalize_3d(arr):
        """
        subtracts mean by each record for each column
        Parameters
        ----------
        arr: ndarray

        Returns
        -------
        out : ndarray
        """
        out = np.zeros_like(arr)
        out[:] = arr[:]
        for i in range(arr.shape[2]):
            np_arr = arr[:, :, i]
            mean_arr = np.mean(np_arr, 1)
            np_arr = np_arr - mean_arr[:, np.newaxis]
            out[:, :, i] = np_arr
        return out

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y=None, **fit_params)
        return self.transform(X, y=None)

    def transform(self, X, y=None):
        return self.normalize_3d(X)


