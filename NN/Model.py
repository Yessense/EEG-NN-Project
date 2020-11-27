import numpy as np  # linear algebra
import pandas as pd
import torch  # PyTorch
# Обработка данных
from sklearn import preprocessing
from torch.nn import functional as F

import Classifier as cs
import utils


class Model:

    def __init__(self, model_name, classes):
        print('Класс Model загружается')
        # Имя модели
        self.model_name = model_name

        # Пути к файлам
        self.normalization_folder = '../NN/normalization/'
        self.model_folder = '../NN/model/'
        self.X_mean_file_name = 'X_mean.npy'
        self.X_std_file_name = 'X_std.npy'
        self.X_fft_mean_file_name = 'X_fft_mean.npy'
        self.X_fft_std_file_name = 'X_fft_std.npy'
        self.model_file_extension = '.pth'

        # Константы
        self.raw_ni = 14
        self.fft_ni = 14

        # Классы
        self.classes = classes  # ['back', 'forward', 'left', 'neutral', 'right']
        self.num_classes = len(classes)

        self.load_normalization_arrays()
        self.load_model()
        print('Загрузка завершена')

    def load_normalization_arrays(self):
        # Загрузка массивов для нормализации
        path_to_folder = self.normalization_folder + self.model_name + '/'

        self.X_mean = np.load(path_to_folder + self.X_mean_file_name)
        self.X_std = np.load(path_to_folder + self.X_std_file_name)
        self.X_fft_mean = np.load(path_to_folder + self.X_fft_mean_file_name)
        self.X_fft_std = np.load(path_to_folder + self.X_fft_std_file_name)

    def load_model(self):
        # Загрузка модели
        self.model = cs.Classifier(self.raw_ni, self.fft_ni, self.num_classes)
        self.model.load_state_dict(torch.load(self.model_folder + self.model_name + self.model_file_extension))
        self.model.eval()

    def process_data(self, X):
        """
        Функция обрабатывает одну секунду данных
        :param X: 128 записей по одной секунде.
        :return: int: class
        """

        # Транспонируем, чтобы смотреть на каналы
        X = X[:, 2:].transpose(1, 0)

        # Нормируем, чтобы значения находились в интервале до единицы
        X = preprocessing.normalize(X, norm='max', axis=1)

        # Создаем признаки при помощи трансформации Фурье
        X_fft = utils.create_fft(X)

        # Нормализуем данные
        X = utils.normalize(X, self.X_mean, self.X_std)
        X_fft = utils.normalize(X_fft, self.X_fft_mean, self.X_fft_std)

        X_t = torch.tensor(X).float()
        X_fft_t = torch.tensor(X_fft).float()

        # Прогоняем данные через модель
        out = self.model.forward(X_t.unsqueeze_(0), X_fft_t.unsqueeze_(0))
        preds = F.log_softmax(out, dim=1).argmax(dim=1)

        return self.classes[preds.item()]


if __name__ == '__main__':
    # Тест на первую запись в файле
    print('Начало теста')
    n = 0  # Номер записи для проверки
    data = pd.read_csv('data.csv')
    test = data.iloc[n * 128: n * 128 + 128].values

    # Пример использования
    mdl = Model('face', ['back', 'forward', 'left', 'neutral', 'right'])
    result = mdl.process_data(test)

    print('Значение в данных: ', data.iloc[n * 128, 0])
    print('Значение выданное нейросетью', result)
