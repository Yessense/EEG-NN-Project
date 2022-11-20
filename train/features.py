import numpy as np
from scipy.stats import entropy
from sklearn.base import BaseEstimator, TransformerMixin
import pywt



class Quantiles(BaseEstimator, TransformerMixin):
    def __init__(self, quantiles=[.50]):
        self.quantiles = quantiles

    def get_quantiles(self, arr, quantiles):
        """
        function gets quantiles from 3d array and returns 2d array
        parameters
        ----------
        arr : np.ndarray
            3d array where fist dimension its timedelta records
            second dimension its frequency intervals
            third dimension its channelwise data
        quantiles : list
            python list with percentiles
            f.e. [0.25, 0.50, 0.75]

        returns
        -------
        out: ndarray
            2d ndarray with columns - len(quantiles) percentiles for each channel

        """
        out = np.quantile(arr, quantiles, axis=1, interpolation='midpoint')
        out = out.transpose(1, 2, 0).reshape(-1, len(quantiles) * arr.shape[2])
        return out

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.get_quantiles(X, self.quantiles)

    def transform(self, X, y=None):
        return self.get_quantiles(X, self.quantiles)


class STD(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def get_std(self, arr):
        """
        function get std feature from 3d array and convert its into 2d array

        Parameters
        ----------
        arr : np.ndarray
            3d array where fist dimension its timedelta records
            second dimension its frequency intervals
            third dimension its channelwise data

        Returns
        -------
        out : ndarray
        """
        out = np.std(arr, axis=1)
        return out

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.get_std(X)

    def transform(self, X, y=None):
        return self.get_std(X)


class Wavelets(BaseEstimator, TransformerMixin):
    def __init__(self, last_n=None, level=None, wavelet='haar', mode='symmetric'):
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
        self.last_n = last_n

    def get_wavelets(self, arr):
        out = pywt.wavedec(arr, wavelet=self.wavelet, level=self.level, mode=self.mode, axis=1)

        if self.last_n:
            out = out[:self.last_n]
        for coef in out:
            print(coef.shape, end=' ')
        print()
        out = np.concatenate(out, axis=1)
        out = out.reshape(-1, out.shape[1] * out.shape[2])

        return out

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.get_wavelets(X)

    def transform(self, X, y=None):
        return self.get_wavelets(X)


class Entropy(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def get_entropy(self, arr):
        """
        function count entropy in 3d array and return 2d array
        Parameters
        ----------
        arr : np.ndarray
            3d array where fist dimension its timedelta records
            second dimension its frequency intervals
            third dimension its channelwise data

        Returns
        -------
        out : ndarray
        """
        out = entropy(arr, axis=1)
        return out

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        self.transform(X, y)

    def transform(self, X, y=None):
        return self.get_entropy(X)


if __name__ == '__main__':
    import utils
    import classifiers
    import pandas as pd
    import numpy as np
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.pipeline import FeatureUnion

    data_folder = './data/'
    file_name = 'data/emotions.csv'
    SEED = 1
    data = pd.read_csv(data_folder + file_name)
    data['class'], class_dict = utils.encode_column(data['class'])

    train, test = utils.eeg_train_test_split(data.to_numpy(), chunk_size=5 * 128, test_size=0.2, random_state=SEED)
    print('Train.shape:', train.shape)
    print('Test.shape:', test.shape)

    X_test, Y_test = utils.create_x_y(test, dt=128, shift=64, verbose=0)
    X_train, Y_train = utils.create_x_y(train, dt=128, shift=64)

    print('X_test.shape:', X_test.shape)
    print('Y_test.shape:', Y_test.shape)
    print('X_train.shape:', X_train.shape)
    print('Y_train.shape:', Y_train.shape)
    print('\nTesting features:\n')

    std = STD()
    entrop = Entropy()
    quantiles = Quantiles(quantiles=[0.5, 0.25, 0.75])

    test_std = std.fit_transform(X_test)
    print('test_std.shape:', test_std.shape)

    test_entrop = entrop.fit_transform(X_test)
    print('test_entrop.shape:', test_entrop.shape)

    test_quantiles = quantiles.fit_transform(X_test)
    print('test_quantiles.shape:', test_quantiles.shape)

    union = FeatureUnion([
        ('STD', STD()),
        ('Entropy', Entropy()),
        ('Quantiles', Quantiles(quantiles=[0.5])),
    ])

    result = union.fit_transform(X_test)
    print('All in one:', result.shape)
