import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def encode_column(column):
    """

    parameters
    ----------
    column : pd.series
        column where to encode values
    returns
    -------
    out : tuple
        transformed column, dict(class: encoding number)
    """
    le = preprocessing.LabelEncoder()
    le.fit(column)

    classes = list(le.classes_)
    encode = le.transform(classes)

    return le.transform(column), dict(zip(classes, encode))


def roll2d(a, b, dx=1, dy=1):
    """
    rolling 2d window for nd array
    last 2 dimensions

    parameters
    ----------
    a : ndarray
        target array where is needed rolling window
    b : tuple
        window array size-like rolling window
    dx : int
        horizontal step, abscissa, number of columns
    dy : int
        vertical step, ordinate, number of rows

    returns
    -------
    out : ndarray
        returned array has shape 4
        first two dimensions have size depends on last two dimensions target array
    """
    shape = a.shape[:-2] + \
            ((a.shape[-2] - b[-2]) // dy + 1,) + \
            ((a.shape[-1] - b[-1]) // dx + 1,) + \
            b  # sausage-like shape with 2d cross-section
    strides = a.strides[:-2] + \
              (a.strides[-2] * dy,) + \
              (a.strides[-1] * dx,) + \
              a.strides[-2:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def roll(a, b, dx=1):
    """
    Rolling 1d window on array

    Parameters
    ----------
    a : ndarray
    b : ndarray
        rolling 1D window array. Example np.zeros(64)
    dx : step size (horizontal)

    Returns
    -------
    out : ndarray
        target array
    """
    shape = a.shape[:-1] + (int((a.shape[-1] - b.shape[-1]) / dx) + 1,) + b.shape
    strides = a.strides[:-1] + (a.strides[-1] * dx,) + a.strides[-1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def eeg_train_test_split(arr, chunk_size, test_size, random_state=None):
    """
    splits data on train/test datasets
    for consistency we need chunks size of x seconds

    parameters
    ----------
    arr : numpy.ndarray
        array need to split
    chunk_size : int
        chunk size its a number of records made on one record iteration f.e. 5 sec = 5 * 128
    n : int
        number of chunks needed to split for each class
    """
    columns = arr.shape[1]
    # getting class indexes
    y = arr[:len(arr) - chunk_size + 1:chunk_size, 0]

    # rolling window on arr
    rolling_window = chunk_size, columns
    x = roll2d(arr, rolling_window, 1, chunk_size).squeeze()

    # creating helpful indexes for splittig and split
    all_indices = list(range(len(y)))
    train_indices, test_indices = train_test_split(all_indices, test_size=test_size, stratify=y,
                                                   random_state=random_state)

    train, test = x[train_indices], x[test_indices]

    train = train.reshape(-1, columns)
    test = test.reshape(-1, columns)

    train = train[np.argsort(train[:, 0], kind='mergesort')]
    test = test[np.argsort(test[:, 0], kind='mergesort')]
    return train, test


def create_x_y(arr, dt, shift=0, verbose=0):
    """
    Creates X, y data from given array

    Parameters
    ----------
    arr : ndarray
        np.array where data is stored
        FIRST COLUMN = class
        SECOND COLUMN = iter
    dt : int
        delta time for one record
    value_counts : list[int]
        In this list we store count of each class in dataframe
        count of class  0, then count of class 1, etc...
    shift : int, optional
        If needed shift to augment data then you need to specify this parametr
        default value = 0
    verbose: int
        default = 0, use 1 if you need information

    Returns
    -------
    out: tuple
        X, y array
    """
    class_column = 0
    iter_column = 1

    if verbose == 1:
        pass
        # print(arr)

    column_number = arr.shape[1] - 2
    class_change = np.roll((np.convolve(arr[:, class_column], [1, -1], 'same') != 0).astype(np.int32), -1)
    iter_change = np.roll((np.convolve(arr[:, iter_column], [1, -1], 'same') != 0).astype(np.int32), -1)

    if verbose == 1:
        print('class_change:', class_change)
        print('iter_change: ', iter_change)

    class_change = np.convolve(class_change, [1, 1], 'same')
    iter_change = np.convolve(iter_change, [1, 1], 'same')

    if verbose == 1:
        print('class_change:', class_change)
        print('iter_change: ', iter_change)

    change = class_change | iter_change

    if verbose == 1:
        print('change:', change)

    conv = np.sum(roll(change, np.ones(dt), shift), axis=1)
    mask = conv < 2

    if verbose == 1:
        print('mask:', mask)

    rolled = roll2d(arr[:, 2:], (dt, column_number), 1, shift).squeeze()
    x = rolled[mask]

    y = arr[:len(arr) - dt + 1:shift, 0]
    y = y[mask]

    return x, y


if __name__ == "__main__":

    # testing create_x_y
    a = np.arange(128).reshape(-1, 4)
    a[0:16, 0] = 0
    a[16:32, 0] = 1
    a[32:, 0] = 2
    a[:8, 1] = 0
    a[8:16, 1] = 1
    a[16:24, 1] = 2
    a[24:32, 1] = 3
    a[32:, 1] = 4
    # print(a)

    size = 5
    shift = 2

    x, y = create_x_y(a, size, shift, 1)

    print(x.shape)
    print(y.shape)