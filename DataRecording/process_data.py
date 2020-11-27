from NN import Classifier as cs

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Для отображения графиков
import matplotlib.pyplot as plt
import seaborn as sns

# Обработка данных
from sklearn import preprocessing
# from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# PyTorch
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader

seed = 1

model = cs.Classifier()
model.eval()

def absfft(x):
    return np.abs(np.fft.rfft(x))

def proccess_data(X):
    '''
    Функция обрабатывает одну секунду данных
    :param chunk: 128 записей по одной секунде.
    :return:
    '''

    X[2:].transpose(1,0)
    chunk = preprocessing.normalize(X, norm = 'max', axis = 1)

    X_fft = np.copy(chunk)
    X_fft = np.apply_along_axis(absfft,2,X_fft)

data = pd.read_csv('data.csv')
test = data.iloc[0:128].values

print(test)
print(test.shape)