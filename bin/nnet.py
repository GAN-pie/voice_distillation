# coding: utf-8

import pickle

import matplotlib.pyplot as plt
import seaborn

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from keras.models import Model, load_model
from keras.layers import LeakyReLU
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.layers import Dense, BatchNormalization, Dropout
from keras.layers import Activation, Input
from keras.layers import Conv1D, Conv2D, MaxPool1D, MaxPool2D
from keras.layers import Flatten, GlobalAveragePooling2D
from keras.constraints import max_norm
from keras.optimizers import SGD, Adam, Adadelta
from keras.initializers import glorot_normal, normal
from keras.regularizers import l2
from keras.utils import to_categorical
import keras.backend as K
from keras.backend.tensorflow_backend import set_session

import tensorflow as tf
from tensorflow.python.ops import math_ops


class IvectorData:

    @staticmethod
    def read_lst_file(fname):
        with open(fname, "r") as fd:
            return [line.strip() for line in fd]

    @staticmethod
    def read_ivector_file(fname):
        with open(fname, "r") as fd:
            data = {}
            for line in fd:
                line = line.strip()
                arr = line.split(" ")
                data[arr[0]] = np.array(arr[1:], dtype=float)
            return data

    def __init__(self, seg_lst_path,
                 feat_file_path, dims=400, field_separator=","):
        self.dims = dims
        self._segments = IvectorData.read_lst_file(seg_lst_path)
        self._ivectors = IvectorData.read_ivector_file(feat_file_path)

        self.ivectors = {k: self._ivectors[k] for k in self._segments}

        if field_separator in [",", "."]:
            self._characters_name = list(map(
                lambda x: x.split(field_separator)[1], self.ivectors.keys()))
        elif field_separator == "-":
            self._characters_name = list(map(
                lambda x: x.split(field_separator)[0], self.ivectors.keys()))

        self._le = LabelEncoder()
        self._labels = self._le.fit_transform(self._characters_name)
        self.num_classes = len(self._le.classes_)

    @property
    def X(self):
        return np.expand_dims(
            pd.DataFrame.from_dict(self.ivectors, orient="index").values,
            axis=-1
        )

    @property
    def Y(self):
        return to_categorical(self._labels, num_classes=self.num_classes)


class LogitData(IvectorData):

    def __init__(self, seg_lst_path, feat_file_path, prior_file_path,
                 dims=400, separator=","):
        super().__init__(
            seg_lst_path, feat_file_path, dims, field_separator=separator)

        self._priors = IvectorData.read_ivector_file(prior_file_path)
        self.priors = {k: self._priors[k] for k in self._segments}
        self.num_classes = len(next(iter(self.priors.values())))

    @property
    def Y(self):
        return pd.DataFrame.from_dict(self.priors, orient="index").values


def custom_softmax(t):
    def activation_fn(x, axis=-1):
        # ndim = K.ndim(x)
        # if ndim == 2:
        #     return K.softmax(x)
        # elif ndim > 2:
        #     e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        #     s = K.sum(e, axis=axis, keepdims=True)
        #     return e / s
        # else:
        #     raise ValueError('Cannot apply softmax to a tensor that is 1D. '
        #                      'Received input: %s' % x)
        x_ = x - K.max(x, axis=axis, keepdims=True)
        e = K.exp(x_ / t)
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    return activation_fn


class Model1D:

    def _make_core_network(self):
        input_layer = Input(shape=self.input_shape)

        x = Conv1D(32, 10)(input_layer)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPool1D()(x)
        x = Dropout(self.drop_rate)(x)

        x = Conv1D(64, 7)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPool1D()(x)
        x = Dropout(self.drop_rate)(x)

        x = Conv1D(256, 4)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPool1D()(x)
        x = Dropout(self.drop_rate)(x)

        x = Conv1D(256, 4)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPool1D()(x)
        x = Dropout(self.drop_rate)(x)

        x = Flatten()(x)

        x = Dense(2048)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(self.drop_rate)(x)

        x = Dense(self.embedding_dim)(x)
        x = BatchNormalization()(x)
        x = Activation("tanh")(x)
        x = Dropout(self.drop_rate)(x)
        return Model(inputs=input_layer, outputs=x, name="core_model")

    def __init__(self, num_classes,
                 embedding_dim=400, drop_rate=0.25,
                 temperature=1, input_shape=(400, 1)):
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.temperature = temperature
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim

        self._W_init = glorot_normal()
        self._b_init = normal(0.5, 10e-2)
        self._W_reg = l2(10e-3)
        self._optimizer = Adadelta()

        self.core_model = self._make_core_network()

        self.input_layer = Input(shape=self.input_shape, name="input")
        self.embedding_layer = self.core_model(self.input_layer)

        if self.temperature == 1:
            self.softmax_layer = Dense(self.num_classes, activation="softmax",
                                       name="softmax")(self.embedding_layer)
        else:
            self.softmax_layer = Dense(
                self.num_classes, activation=custom_softmax(self.temperature),
                name="softmax")(self.embedding_layer)

        self.classifier_model = Model(
            self.input_layer,
            self.softmax_layer,
            name="classifier_model"
        )

        self.classifier_model.compile(
            self._optimizer, "categorical_crossentropy", metrics=["accuracy"])
        self.classifier_model.summary()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        set_session(sess)


class Model1DStudent(Model1D):
    def _make_core_network(self):
        input_layer = Input(shape=self.input_shape)

        x = Conv1D(32, 10)(input_layer)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPool1D()(x)
        x = Dropout(self.drop_rate)(x)

        x = Conv1D(64, 7)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPool1D()(x)
        x = Dropout(self.drop_rate)(x)

        x = Conv1D(128, 4)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPool1D()(x)
        x = Dropout(self.drop_rate)(x)

        x = Flatten()(x)

        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(self.drop_rate)(x)

        x = Dense(self.embedding_dim)(x)
        x = BatchNormalization()(x)
        x = Activation("tanh")(x)
        x = Dropout(self.drop_rate)(x)
        return Model(inputs=input_layer, outputs=x, name="core_model")


class Model2D:
    def __init__(self, num_classes, embedding_dim=400,
                 drop_rate=0.25, temperature=1,
                 input_shape=(None, None, 1)):
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim

        self._W_init = glorot_normal()
        self._b_init = normal(0.5, 10e-2)
        self._W_reg = l2(10e-3)
        self._optimizer = Adadelta()

        self.core_model = self._make_core_network()

        self.input_layer = Input(self.input_shape, name="input")
        self.embedding_layer = self.core_model(self.input_layer)
        self.softmax_layer = Dense(
            self.num_classes, activation="softmax")(self.embedding_layer)
        self.classifier_model = Model(self.input_layer, self.softmax_layer)

        self.classifier_model.compile(
            self._optimizer, "categorical_crossentropy", metrics=["accuracy"])
        self.classifier_model.summary()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        set_session(sess)

    def _make_core_network(self):
        input_layer = Input(shape=self.input_shape)

        x = Conv2D(32, (10, 10))(input_layer)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPool2D()(x)
        x = Dropout(self.drop_rate)(x)

        x = Conv2D(64, (7, 7))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPool2D()(x)
        x = Dropout(self.drop_rate)(x)

        x = Conv2D(256, (4, 4))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPool2D()(x)
        x = Dropout(self.drop_rate)(x)

        x = Conv2D(256, (4, 4))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPool2D()(x)
        x = Dropout(self.drop_rate)(x)

        x = GlobalAveragePooling2D()(x)

        x = Dense(2048)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(self.drop_rate)(x)

        x = Dense(self.embedding_dim)(x)
        x = BatchNormalization()(x)
        x = Activation("tanh")(x)
        x = Dropout(self.drop_rate)(x)

        return Model(inputs=input_layer, outputs=x, name="core_model")


class TrainHistory(Callback):
    def __init__(self, logs={}):
        self.losses = []
        self.accuracies = []
        self.validation_losses = []
        self.validation_accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get("loss"))
        self.accuracies.append(logs.get("acc"))

    def on_epoch_end(self, epoch, logs={}):
        self.validation_losses.append(logs.get("val_loss"))
        self.validation_accuracies.append(logs.get("val_acc"))

    @staticmethod
    def save(obj, save_path):
        with open(save_path, "wb") as fd:
            pickler = pickle.Pickler(fd)
            pickler.dump({
                "acc": obj.accuracies, "val_acc": obj.validation_accuracies,
                "loss": obj.losses, "val_loss": obj.validation_losses
            })

    def plot_history(self, plot_path):
        (fig, axes) = plt.subplots(2, 2)
        axes[0][0].plot(self.losses)
        axes[0][0].set_ylabel("loss")
        axes[0][0].set_xlabel("iterations")
        axes[0][1].plot(self.validation_losses)
        axes[0][1].set_ylabel("validation loss")
        axes[0][1].set_xlabel("epochs")
        axes[1][0].plot(self.accuracies)
        axes[1][0].set_ylabel("accuracy")
        axes[1][0].set_xlabel("iterations")
        axes[1][1].plot(self.validation_accuracies)
        axes[1][1].set_ylabel("validation_accuracy")
        axes[1][1].set_xlabel("epochs")

        plt.tight_layout()

        plt.savefig(plot_path)
        plt.close()


class Monitor(Callback):
    def on_batch_end(self, batch, logs={}):
        print(
            "iter #{0:5d} - loss={1:.4f} - acc={2:.4f}".format(
                batch + 1, logs.get("loss"), logs.get("acc")
            ),
            end="\r", flush=True
        )

    def on_epoch_end(self, epoch, logs={}):
        print(
            "\nepoch #{0:4d} - loss={1:.4f} - acc={2:.4f} - \
            val_loss={3:.4f} - val_acc={4:.4f}".format(
                epoch+1,
                logs.get("loss"),
                logs.get("acc"),
                logs.get("val_loss"),
                logs.get("val_acc")
            ),
            flush=True
        )
