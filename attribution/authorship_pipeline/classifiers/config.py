from typing import Union

import yaml


class Config:

    def __init__(self, config):
        self.config = config

    @classmethod
    def fromyaml(cls, filename: str):
        return cls(yaml.load(open(filename, 'r'), Loader=yaml.Loader))

    def classifier_type(self):
        return self.__get('classifier_type')

    def source_folder(self):
        return self.__get('source_folder')

    def features(self):
        return self.__get('features')

    def feature_count(self):
        return self.__get('feature_count')

    def seed(self):
        return self.__get('seed')

    def entities(self):
        return self.__get('entities')

    def n_classes(self):
        return self.__get('n_classes')

    def test_size(self) -> Union[int, float]:
        return self.__get('test_size')

    def params(self):
        return self.__get('params')

    def n_runs(self):
        return self.__get('n_runs')

    def mutual_info_file(self):
        return self.__get('mutual_info_file')

    def epochs(self):
        return self.__get('epochs')

    def batch_size(self):
        return self.__get('batch_size')

    def hidden_dim(self):
        return self.__get('hidden_dim')

    def learning_rate(self):
        return self.__get('learning_rate')

    def log_batches(self):
        return self.__get('log_batches')

    def time_folds(self):
        return self.__get('time_folds')

    def mode(self):
        return self.__get('mode')

    def min_max_count(self):
        min_count = self.__get('min_count')
        if min_count is None:
            min_count = 100
        max_count = self.__get('max_count')
        if max_count is None:
            max_count = 10 ** 9
        return min_count, max_count

    def min_max_train(self):
        min_train = self.__get('min_train')
        if min_train is None:
            min_train = 0.7
        max_train = self.__get('max_train')
        if max_train is None:
            max_train = 0.8
        return min_train, max_train

    def __get(self, param):
        try:
            return self.config[param]
        except KeyError:
            return None
