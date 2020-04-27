# coding: utf-8

import pickle

import pandas as pd
import numpy as np
from keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def read_lst_file(fname):
    with open(fname, "r") as fd:
        return [line.strip() for line in fd]


def read_data_file(fname):
    with open(fname, "r") as fd:
        data = {}
        for line in fd:
            line = line.strip()
            arr = line.split(" ")
            data[arr[0]] = np.array(arr[1:], dtype=float)
        return data


class IvectorMassEffect:

    def __init__(self, lst_path, feat_path, dims=400, expand=False, scaler=None):
        self.input_dim = dims
        self._expand = expand
        self._segments = read_lst_file(lst_path)
        self._ivectors = read_data_file(feat_path)

        if scaler:
            with open(scaler, "rb") as fd:
                self.scaler = pickle.Unpickler(fd).load()
        else:
            self.scaler = None

        self.ivectors = {k: self._ivectors[k] for k in self._segments}

        self._labels = list(map(lambda x: x.split(",")[1], self.ivectors.keys()))

        self._le = LabelEncoder()
        self.labels = self._le.fit_transform(self._labels)
        self.num_classes = len(self._le.classes_)

    @property
    def X(self):
        array = pd.DataFrame.from_dict(self.ivectors, orient="index").values

        if self.scaler is None:
            self.scaler = StandardScaler()
            array = self.scaler.fit_transform(array)
        else:
            array = self.scaler.transform(array)

        if not self._expand:
            return array
        else:
            return np.expand_dims(array, axis=-1)

    @property
    def Y(self):
        return to_categorical(self.labels, num_classes=self.num_classes)

    def save_scaler(self, fname):
        with open(fname, "wb") as fd:
            pickle.Pickler(fd).dump(self.scaler)


class IvectorSkyrim:

    def __init__(self, lst_path, feat_path, dims=400, expand=False, scaler=None):
        self.input_dim = dims
        self._expand = expand
        self._segments = read_lst_file(lst_path)
        self._ivectors = read_data_file(feat_path)

        if scaler:
            with open(scaler, "rb") as fd:
                self.scaler = pickle.Unpickler(fd).load()
        else:
            self.scaler = None

        self.ivectors = {k: self._ivectors[k] for k in self._segments}

        self._labels = list(map(lambda x: x.split(".")[1], self.ivectors.keys()))

        self._le = LabelEncoder()
        self.labels = self._le.fit_transform(self._labels)
        self.num_classes = len(self._le.classes_)
        self.scaler = None

    @property
    def X(self):
        array = pd.DataFrame.from_dict(self.ivectors, orient="index").values

        if self.scaler is None:
            self.scaler = StandardScaler()
            array = self.scaler.fit_transform(array)
        else:
            array = self.scaler.transform(array)
        if not self._expand:
            return array
        else:
            return np.expand_dims(array, axis=-1)

    @property
    def Y(self):
        return to_categorical(self.labels, num_classes=self.num_classes)

    def save_scaler(self, fname):
        with open(fname, "wb") as fd:
            pickle.Pickler(fd).dump(self.scaler)

class IvectorVoxceleb:

    def __init__(self, lst_path, feat_path, dims=400, expand=False, scaler=None):
        self.input_dim = dims
        self._expand = expand
        self._segments = read_lst_file(lst_path)
        self._ivectors = read_data_file(feat_path)

        if scaler:
            with open(scaler, "rb") as fd:
                self.scaler = pickle.Unpickler(fd).load()
        else:
            self.scaler = None

        self.ivectors = {k: self._ivectors[k] for k in self._segments}

        self._labels = list(map(lambda x: x.split("-")[0], self.ivectors.keys()))

        self._le = LabelEncoder()
        self.labels = self._le.fit_transform(self._labels)
        self.num_classes = len(self._le.classes_)
        self.scaler = None

    @property
    def X(self):
        array = pd.DataFrame.from_dict(self.ivectors, orient="index").values

        if self.scaler is None:
            self.scaler = StandardScaler()
            array = self.scaler.fit_transform(array)
        else:
            array = self.scaler.transform(array)
        if not self._expand:
            return array
        else:
            return np.expand_dims(array, axis=-1)

    @property
    def Y(self):
        return to_categorical(self.labels, num_classes=self.num_classes)

    def save_scaler(self, fname):
        with open(fname, "wb") as fd:
            pickle.Pickler(fd).dump(self.scaler)



class IvectorSoftTargetSkyrim():

    def __init__(self, lst_path, feat_path, soft_path, dims=400, expand=False):
        self.input_dim = dims
        self._expand = expand
        self._segments = read_lst_file(lst_path)
        self._ivectors = read_data_file(feat_path)

        self.ivectors = {k: self._ivectors[k] for k in self._segments}

        self._soft_targets = read_data_file(soft_path)
        self.soft_targets = {k: self._soft_targets[k] for k in self.ivectors.keys()}

        self._labels = list(map(lambda x: x.split(".")[1], self.ivectors.keys()))

        self._le = LabelEncoder()
        self.labels = self._le.fit_transform(self._labels)

        self.num_classes = len(self.soft_targets[self._segments[0]])
        self.num_hard_classes = len(self._le.classes_)
        self.scaler = None

    @property
    def X(self):
        array = pd.DataFrame.from_dict(self.ivectors, orient="index").values
        if self.scaler is None:
            self.scaler = StandardScaler()
            array = self.scaler.fit_transform(array)
        else:
            array = self.scaler.transform(array)
        if not self._expand:
            return array
        else:
            return np.expand_dims(array, axis=-1)

    @property
    def Y(self):
        return pd.DataFrame.from_dict(self.soft_targets, orient="index").values

    @property
    def Y_hard(self):
        return to_categorical(self.labels, num_classes=self.num_hard_classes)

    def save_scaler(self, fname):
        with open(fname, "wb") as fd:
            pickle.Pickler(fd).dump(self.scaler)


class IvectorSoftTargetMassEffect():

    def __init__(self, lst_path, feat_path, soft_path, dims=400, expand=False):
        self.input_dim = dims
        self._expand = expand
        self._segments = read_lst_file(lst_path)
        self._ivectors = read_data_file(feat_path)

        self.ivectors = {k: self._ivectors[k] for k in self._segments}

        self._soft_targets = read_data_file(soft_path)
        self.soft_targets = {k: self._soft_targets[k] for k in self.ivectors.keys()}

        self._labels = list(map(lambda x: x.split(",")[1], self.ivectors.keys()))

        self._le = LabelEncoder()
        self.labels = self._le.fit_transform(self._labels)

        self.num_classes = len(self.soft_targets[self._segments[0]])
        self.num_hard_classes = len(self._le.classes_)
        self.scaler = None

    @property
    def X(self):
        array = pd.DataFrame.from_dict(self.ivectors, orient="index").values
        if self.scaler is None:
            self.scaler = StandardScaler()
            array = self.scaler.fit_transform(array)
        else:
            array = self.scaler.transform(array)
        if not self._expand:
            return array
        else:
            return np.expand_dims(array, axis=-1)

    @property
    def Y(self):
        return pd.DataFrame.from_dict(self.soft_targets, orient="index").values

    @property
    def Y_hard(self):
        return to_categorical(self.labels, num_classes=self.num_hard_classes)

    def save_scaler(self, fname):
        with open(fname, "wb") as fd:
            pickle.Pickler(fd).dump(self.scaler)


class IvectorSoftTargetVoxceleb:
    def __init__(self, lst_path, feat_path, soft_path, dims=400, expand=False, scaler=None):
        self.input_dim = dims
        self._expand = expand
        self._segments = read_lst_file(lst_path)
        self._ivectors = read_data_file(feat_path)

        if scaler:
            with open(scaler, "rb") as fd:
                self.scaler = pickle.Unpickler(fd).load()
        else:
            self.scaler = None

        self.ivectors = {k: self._ivectors[k] for k in self._segments}

        self._soft_targets = read_data_file(soft_path)
        self.soft_targets = {k: self._soft_targets[k] for k in self.ivectors.keys()}

        self._labels = list(map(lambda x: x.split("-")[0], self.ivectors.keys()))

        self._le = LabelEncoder()
        self.labels = self._le.fit_transform(self._labels)
        self.num_classes = len(self._le.classes_)

        self._array = pd.DataFrame.from_dict(self.ivectors, orient="index").values
        self._array_Y = pd.DataFrame.from_dict(self.soft_targets, orient="index").values
        self._X, self._X_test, self._Y, self._Y_test = train_test_split(self._array, self._array_Y, test_size=0.2)

    @property
    def X(self):
        array = self._X

        if self.scaler is None:
            self.scaler = StandardScaler()
            array = self.scaler.fit_transform(self._X)
        else:
            array = self.scaler.transform(self._X)

        if not self._expand:
            return array
        else:
            return np.expand_dims(array, axis=-1)

    @property
    def X_test(self):
        array = self._X

        if self.scaler is None:
            self.scaler = StandardScaler()
            array = self.scaler.fit_transform(self._X)
        else:
            array = self.scaler.transform(self._X)

        if not self._expand:
            return array
        else:
            return np.expand_dims(array, axis=-1)

    @property
    def Y(self):
        return self._Y

    @property
    def Y_test(self):
        return self._Y_test

    def save_scaler(self, fname):
        with open(fname, "wb") as fd:
            pickle.Pickler(fd).dump(self.scaler)
