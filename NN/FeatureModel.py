import feature_utils
from joblib import dump, load
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


class FeatureModel:
    def __init__(self, model_name, classes):
        self.name = model_name
        self.classes = classes
        self.path_to_folder = '../NN/feature_model/'

        self.load_transformer()
        self.load_classifier()

    def load_transformer(self):
        preprocess = load(self.path_to_folder + 'preprocessing.joblib')
        feature_extraction = load(self.path_to_folder + 'features_extraction.joblib')
        self.transformer = Pipeline(steps=[
            ('preprocess', preprocess),
            ('feature_extraction', feature_extraction),
        ])

    def load_classifier(self):
        self.classifier = load(self.path_to_folder + 'LGBMClassifier.joblib')

    def get_class(self, x):
        return self.classes[round(x[0])]

    def process_data(self, X):
        out = X[np.newaxis, :, 2:]
        print(out.shape)
        out = self.transformer.transform(out)
        print(out.shape)
        out = self.classifier.predict(out)
        print(out.shape)
        out = self.get_class(out)

        return out


if __name__ == '__main__':
    # Тест на первую запись в файле
    print('Начало теста')
    n = 8  # Номер записи для проверки
    data = pd.read_csv('../DataRecording/data.csv')
    test = data.iloc[n * 8 * 128: n * 8 * 128 + 128].values


    # Пример использования
    mdl = FeatureModel('face', ['concentrate', 'relax'])
    print(mdl.classes)
    print(test.shape)
    result = mdl.process_data(test)

    print('Значение в данных: ', data.iloc[n * 8* 128, 0])
    print('Значение выданное нейросетью', result)
