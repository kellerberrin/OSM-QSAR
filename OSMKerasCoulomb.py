# MIT License
#
# Copyright (c) 2017 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#
# Python 2 and Python 3 compatibility imports.
from __future__ import absolute_import, division, print_function, unicode_literals
from six import with_metaclass
import copy

import sys
import os

import numpy as np

from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.layers.noise import GaussianDropout, GaussianNoise
from keras.layers import normalization, BatchNormalization
from keras.regularizers import l2, l1_l2
from keras.models import load_model
from keras.constraints import maxnorm
from keras.initializers import TruncatedNormal
from keras.optimizers import SGD, Adam, Adagrad, Adadelta
from keras.utils import np_utils

#from keras.utils.visualize_util import plot
import keras.backend as backend

from OSMBase import ModelMetaClass  # The virtual model class.
from OSMKerasBase import KlassSequential
from OSMModelData import OSMModelData
from OSMAnalysis import OSMSensitivity, OSMDragonSensitivity




# ===============================================================================
# Keras Pattern Classifier, fits ION ACTIVITY to a Coulomb Matrix.
# ===============================================================================

class CoulombMatrix(with_metaclass(ModelMetaClass, KlassSequential)):
    def __init__(self, args, log):
        super(CoulombMatrix, self).__init__(args, log)

        # Define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = {"DEPENDENT": {"VARIABLE": "ION_ACTIVITY", "SHAPE": [3], "TYPE": OSMModelData.CLASSES}
            , "INDEPENDENT": [{"VARIABLE": "COULOMB_ARRAY", "SHAPE": [2916], "TYPE": OSMModelData.FLOAT64}]}

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "COULOMB > ION_ACTIVITY Classifier"

    def model_postfix(self):  # Must be unique for each model.
        return "ion_c"

    def model_description(self):
        return ("A KERAS (TensorFlow) multi-layer Neural Network class classification model. \n"
                "This classifier analyzes molecular coulomb matrices against ION_ACTIVITY")


    def model_define(self):  # Defines the modified sequential class with regularizers defined.

        model = Sequential()
        l2_param = 0.0
        l1_param = 0.0
        dropout_param = 0.5
        Gaussian_noise = 1
        initializer = "uniform"
        activation_func = "relu"

        adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=5e-09)

        model.add(Dense(2048, input_dim=2916, kernel_initializer=initializer, activation=activation_func
                        , activity_regularizer=l1_l2(l1_param, l2_param), kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(GaussianNoise(Gaussian_noise))

        model.add(Dense(2048, kernel_initializer=initializer, activation=activation_func
                        , activity_regularizer=l1_l2(l1_param, l2_param), kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(GaussianNoise(Gaussian_noise))

        model.add(Dense(2048, kernel_initializer=initializer, activation=activation_func
                        , activity_regularizer=l1_l2(l1_param, l2_param), kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(GaussianNoise(Gaussian_noise))

        model.add(Dense(1024, kernel_initializer=initializer, activation=activation_func
                        , activity_regularizer=l1_l2(l1_param, l2_param), kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(GaussianNoise(Gaussian_noise))

        model.add(Dense(512, kernel_initializer=initializer, activation=activation_func
                        , activity_regularizer=l1_l2(l1_param, l2_param), kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(GaussianNoise(Gaussian_noise))

        model.add(Dense(64, kernel_initializer=initializer, activation=activation_func
                        , activity_regularizer=l1_l2(l1_param, l2_param), kernel_constraint=maxnorm(3)))

        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())

        model.add(Dense(3, activation="softmax", kernel_initializer=initializer))
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

        return model

# ===============================================================================
# Keras Pattern Classifier, fits ION ACTIVITY to a Convolved Coulomb Matrix.
# ===============================================================================

class CoulombConvolution(with_metaclass(ModelMetaClass, KlassSequential)):
    def __init__(self, args, log):
        super(CoulombConvolution, self).__init__(args, log)

        # Define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = {"DEPENDENT": {"VARIABLE": "ION_ACTIVITY", "SHAPE": [3], "TYPE": OSMModelData.CLASSES}
            , "INDEPENDENT": [{"VARIABLE": "COULOMB", "SHAPE": None, "TYPE": OSMModelData.FLOAT64}]}

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "COULOMB > ION_ACTIVITY Convolution Classifier"

    def model_postfix(self):  # Must be unique for each model.
        return "ion_cc"

    def model_description(self):
        return ("A KERAS (TensorFlow) multi-layer Neural Network class classification model. \n"
                "This classifier analyzes convolved molecular coulomb matrices against ION_ACTIVITY")

    def model_define(self):  # Defines the modified sequential class with regularizers defined.

        model = Sequential()
        l2_param = 0.0
        l1_param = 0.0
        dropout_param = 0.2
        Gaussian_noise = 1
        initializer = "uniform"
        activation_func = "relu"
        kernel_size = (4,4)
        pool_size = (2,2)
        filters = 32

        input_tensor_shape = self.data.training().input_data().shape
        input_shape = tuple(x for i, x in enumerate(input_tensor_shape) if i > 0) # remove rows

        info_text = "The 'ion_cc' input tensor shape is {}".format(input_tensor_shape)
        self.log.info(info_text)

        adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=5e-09)

        model.add(Conv2D(filters=filters, kernel_size=kernel_size, input_shape=input_shape, activation=activation_func
                         ,data_format='channels_first', kernel_initializer=initializer))
        model.add(MaxPooling2D(pool_size=pool_size))

        model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation=activation_func
                         , kernel_initializer=initializer))
        model.add(MaxPooling2D(pool_size=(3,3)))

        model.add(Flatten())

        model.add(Dense(512, activation=activation_func, kernel_constraint=maxnorm(3)
                        ,kernel_initializer=initializer))
        model.add(Dropout(dropout_param))

        model.add(Dense(512, activation=activation_func, kernel_constraint=maxnorm(3)
                        ,kernel_initializer=initializer))
        model.add(Dropout(dropout_param))

        model.add(Dense(3, activation='softmax'))

        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

        return model

# ===============================================================================
# Keras Pattern Classifier, fits ION ACTIVITY to the Coulomb Eigen values data.
# ===============================================================================

class CoulombEigen(with_metaclass(ModelMetaClass, KlassSequential)):
    def __init__(self, args, log):
        super(CoulombEigen, self).__init__(args, log)

        # Define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = {"DEPENDENT": {"VARIABLE": "ION_ACTIVITY", "SHAPE": [3], "TYPE": OSMModelData.CLASSES}
            , "INDEPENDENT": [{"VARIABLE": "COULOMB_EIGEN", "SHAPE": [54], "TYPE": OSMModelData.FLOAT64}]}

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "COULOMB_EIGEN > ION_ACTIVITY Classifier"

    def model_postfix(self):  # Must be unique for each model.
        return "ion_e"

    def model_description(self):
        return ("A KERAS (TensorFlow) multi-layer Neural Network class classification model. \n"
                "This classifier analyzes molecular coulomb matrix eigenvalues against ION_ACTIVITY")

    def model_define(self):  # Defines the modified sequential class with regularizers defined.

        model = Sequential()
        l2_param = 0.0
        l1_param = 0.0
        dropout_param = 0.5
        Gaussian_noise = 1
        initializer = "uniform"
        activation_func = "relu"

        adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=5e-09)

        model.add(Dense(256, input_dim=54, kernel_initializer=initializer, activation=activation_func
                        , activity_regularizer=l1_l2(l1_param, l2_param), kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(GaussianNoise(Gaussian_noise))

        model.add(Dense(1024, kernel_initializer=initializer, activation=activation_func
                        , activity_regularizer=l1_l2(l1_param, l2_param), kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(GaussianNoise(Gaussian_noise))

        model.add(Dense(1024, kernel_initializer=initializer, activation=activation_func
                        , activity_regularizer=l1_l2(l1_param, l2_param), kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(GaussianNoise(Gaussian_noise))

        model.add(Dense(1024, kernel_initializer=initializer, activation=activation_func
                        , activity_regularizer=l1_l2(l1_param, l2_param), kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(GaussianNoise(Gaussian_noise))

        model.add(Dense(512, kernel_initializer=initializer, activation=activation_func
                        , activity_regularizer=l1_l2(l1_param, l2_param), kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(GaussianNoise(Gaussian_noise))

        model.add(Dense(64, kernel_initializer=initializer, activation=activation_func
                        , activity_regularizer=l1_l2(l1_param, l2_param), kernel_constraint=maxnorm(3)))

        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())

        model.add(Dense(3, activation="softmax", kernel_initializer=initializer))
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

        return model



