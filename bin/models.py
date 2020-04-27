# coding: utf-8

import sys
import os
from os import path

import numpy as np
import keras

from keras import backend as K
from keras.layers import Input, Dense, Activation, AlphaDropout, Dropout
from keras.optimizers import SGD, Adadelta
from keras.initializers import lecun_normal, glorot_normal
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf


def custom_softmax(T):
    def activation_fn(w, axis=-1):
        e = K.exp(w / T)
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    return activation_fn


class ModelTeacher:

    def __init__(self, num_classes, input_shape=(400,), embedding_dim=128, T=1.0, drop_rate=0.25):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.T = T
        self.drop_rate = drop_rate

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)
        set_session(sess)

        self._W_init = glorot_normal()

        self.input_layer = Input(shape=self.input_shape)

        self.core = Dense(256, kernel_initializer=self._W_init)(self.input_layer)
        self.core = Activation("tanh")(self.core)
        self.core = Dropout(self.drop_rate)(self.core)

        self.core = Dense(256, kernel_initializer=self._W_init)(self.core)
        self.core = Activation("tanh")(self.core)
        self.core = Dropout(self.drop_rate)(self.core)

        self.core = Dense(256, kernel_initializer=self._W_init)(self.core)
        self.core = Activation("tanh")(self.core)
        self.core = Dropout(self.drop_rate)(self.core)

        self.core = Dense(self.embedding_dim, kernel_initializer=self._W_init, name="embedding")(self.core)
        self.core = Activation("tanh")(self.core)
        self.core = Dropout(2 * self.drop_rate)(self.core)

        self.softmax_layer = Dense(self.num_classes, activation=custom_softmax(
            self.T), name="softmax")(self.core)

        self.model = Model(self.input_layer, self.softmax_layer, name="teacher")
        # self._opti = Adadelta(clipnorm=1.0)
        # self.model.compile(self._opti, "categorical_crossentropy", metrics=["accuracy"])
        self.model.compile("adadelta", "categorical_crossentropy", metrics=["accuracy"])
        self.model.summary()

    def save(self, out_dir):
        self.model.save(path.join(out_dir, "teacher_model.h5"))


class ModelStudent:

    def __init__(self, num_classes, input_shape=(400,), embedding_dim=128, T=1.0, drop_rate=0.25):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.T = T
        self.drop_rate = drop_rate

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)
        set_session(sess)

        self._W_init = glorot_normal()

        self.input_layer = Input(shape=self.input_shape)

        self.core = Dense(256, kernel_initializer=self._W_init)(self.input_layer)
        self.core = Activation("tanh")(self.core)
        self.core = Dropout(self.drop_rate)(self.core)

        self.core = Dense(256, kernel_initializer=self._W_init)(self.core)
        self.core = Activation("tanh")(self.core)
        self.core = Dropout(self.drop_rate)(self.core)

        self.core = Dense(256, kernel_initializer=self._W_init)(self.core)
        self.core = Activation("tanh")(self.core)
        self.core = Dropout(self.drop_rate)(self.core)

        self.core = Dense(self.embedding_dim, kernel_initializer=self._W_init, name="embedding")(self.core)
        self.core = Activation("tanh")(self.core)
        self.core = Dropout(2 * self.drop_rate)(self.core)

        self.softmax_layer = Dense(self.num_classes, activation=custom_softmax(
            self.T), name="softmax")(self.core)

        self.model = Model(self.input_layer, self.softmax_layer, name="student")
        # self._opti = Adadelta(clipnorm=1.0)
        # self.model.compile(self._opti, "categorical_crossentropy", metrics=["accuracy"])
        self.model.compile("adadelta", "categorical_crossentropy", metrics=["accuracy"])
        self.model.summary()

    def save(self, out_dir):
        # self.core.save_weights(path.join(out_dir, "teacher_core_weights.h5"))
        self.model.save(path.join(out_dir, "student_model.h5"))


class ModelStudentDual:

    def __init__(self, num_classes, num_hard_classes, input_shape=(400,), embedding_dim=128, T=1.0, imitation=0.5, drop_rate=0.25):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.T = T
        self.drop_rate = drop_rate
        self.imitation = imitation
        self.num_hard_classes = num_hard_classes

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)
        set_session(sess)

        self._W_init = glorot_normal()

        self.input_layer = Input(shape=self.input_shape)

        self.core = Dense(256, kernel_initializer=self._W_init)(self.input_layer)
        self.core = Activation("tanh")(self.core)
        self.core = Dropout(self.drop_rate)(self.core)

        self.core = Dense(256, kernel_initializer=self._W_init)(self.core)
        self.core = Activation("tanh")(self.core)
        self.core = Dropout(self.drop_rate)(self.core)

        self.core = Dense(256, kernel_initializer=self._W_init)(self.core)
        self.core = Activation("tanh")(self.core)
        self.core = Dropout(self.drop_rate)(self.core)

        self.core = Dense(self.embedding_dim, kernel_initializer=self._W_init, name="embedding")(self.core)
        self.core = Activation("relu")(self.core)
        self.core = Dropout(2 * self.drop_rate)(self.core)

        self.softmax_layer = Dense(self.num_classes, activation=custom_softmax(
            self.T), name="softmax")(self.core)

        self.hard_softmax_layer = Dense(self.num_hard_classes, activation="softmax", name="hard_softmax")(self.core)

        self.model = Model(inputs=self.input_layer, outputs=[self.softmax_layer, self.hard_softmax_layer], name="student")
        # self._opti = Adadelta(clipnorm=1.0)
        # self.model.compile(self._opti, "categorical_crossentropy", metrics=["accuracy"])
        self.model.compile("adadelta", loss=["categorical_crossentropy", "categorical_crossentropy"],
                           loss_weights=[self.imitation, (1.0 - self.imitation)], metrics=["accuracy"])
        self.model.summary()

    def save(self, out_dir):
        # self.core.save_weights(path.join(out_dir, "teacher_core_weights.h5"))
        self.model.save(path.join(out_dir, "student_model.h5"))
