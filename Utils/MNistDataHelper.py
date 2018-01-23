import pickle
from DataHelper import DatasetHelper
from FileHelper import *
import gzip
import random

class MNistDataHelper(DatasetHelper):
    def _convert_type(self,
                      _variables,
                      _types):
        variables = []
        for _variable, _type in zip(_variables, _types):
            variables.append(_variable.astype(_type))
        return variables

    def __init__(self,
                 _dataset_path = None):
        DatasetHelper.__init__(self)

        # Check parameters
        check_not_none(_dataset_path, '_dataset_path');

        # Set parameters
        self.dataset_path  = _dataset_path

        # Load the dataset
        with gzip.open(self.dataset_path, 'rb') as _file:
            try:
                _train_set, _valid_set, _test_set = pickle.load(_file, encoding='latin1')
            except:
                _train_set, _valid_set, _test_set = pickle.load(_file)

        self.train_set_x, self.train_set_y = _train_set
        self.valid_set_x, self.valid_set_y = _valid_set
        self.test_set_x, self.test_set_y   = _test_set

        self.train_set_x, self.train_set_y = self._convert_type([self.train_set_x, self.train_set_y], ['float32', 'int32'])
        self.valid_set_x, self.valid_set_y = self._convert_type([self.valid_set_x, self.valid_set_y], ['float32', 'int32'])
        self.test_set_x, self.test_set_y   = self._convert_type([self.test_set_x, self.test_set_y], ['float32', 'int32'])