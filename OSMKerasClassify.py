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
from keras.layers import Dropout
from keras.layers import normalization, BatchNormalization
from keras.regularizers import l2, l1l2, activity_l2
from keras.models import load_model
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam, Adagrad, Adadelta
from keras.utils import np_utils

#from keras.utils.visualize_util import plot
import keras.backend as backend

from OSMBase import ModelMetaClass  # The virtual model class.
from OSMKerasBase import KlassSequential
from OSMModelData import OSMModelData


# ===============================================================================
# Keras Pattern Classifier, fits ION ACTIVITY to MACCS FP
# ===============================================================================

class KlassIonMaccs(with_metaclass(ModelMetaClass, KlassSequential)):
    def __init__(self, args, log):
        super(KlassIonMaccs, self).__init__(args, log)

        # Define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = {"DEPENDENT": {"VARIABLE": "ION_ACTIVITY", "SHAPE": [3], "TYPE": OSMModelData.CLASSES}
            , "INDEPENDENT": [{"VARIABLE": "MACCFP", "SHAPE": [167], "TYPE": OSMModelData.FLOAT64}]}

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "MACCS > ION_ACTIVITY Classifier"

    def model_postfix(self):  # Must be unique for each model.
        return "ion_macc"

    def model_description(self):
        return ("A KERAS (TensorFlow) multi-layer Neural Network class classification model. \n"
                "This classifier analyzes the MACCS fingerprint against ION_ACTIVITY")


    def model_define(self):  # Defines the modified sequential class with regularizers defined.

        model = Sequential()
        l2_param = 0.0
        l1_param = 0.0
        dropout_param = 0.0

        adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=5e-09)

        model.add(Dense(2048, input_dim=167, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param, l2_param), W_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(2048, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param,l2_param), W_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(512, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param, l2_param), W_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(64, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param, l2_param), W_constraint=maxnorm(3)))

        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(3, activation = "softmax", init="normal"))
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

        return model


    # ===============================================================================
    # Keras Pattern Classifier, fits ION ACTIVITY to MORGAN2048_4
    # ===============================================================================

class KlassIonMorgan(with_metaclass(ModelMetaClass, KlassSequential)):
    def __init__(self, args, log):
        super(KlassIonMorgan, self).__init__(args, log)

        # Define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = {"DEPENDENT": {"VARIABLE": "ION_ACTIVITY", "SHAPE": [3], "TYPE": OSMModelData.CLASSES}
            , "INDEPENDENT": [{"VARIABLE": "MORGAN2048_4", "SHAPE": [2048], "TYPE": OSMModelData.FLOAT64}]}

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "MORGAN > ION_ACTIVITY Classifier"

    def model_postfix(self):  # Must be unique for each model.
        return "ion_m"

    def model_description(self):

        return ("A KERAS (TensorFlow) multi-layer Neural Network class classification model. \n"
                "This classifier analyzes the MORGAN2048_n, TOPOLOGICAL2048  fingerprints against ION_ACTIVITY")

    def model_define(self):  # Defines the modified sequential class with regularizers defined.

        model = Sequential()
        l2_param = 0.0
        l1_param = 0.0
        dropout_param = 0.0

        adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=5e-09)

        model.add(Dense(2048, input_dim=2048, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param, l2_param), W_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(2048, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param,l2_param), W_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(512, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param, l2_param), W_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(64, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param, l2_param), W_constraint=maxnorm(3)))

        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(3, activation = "softmax", init="normal"))
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

        return model


# ===============================================================================
# Keras Pattern Classifier, fits ION ACTIVITY to MORGAN2048_4
# ===============================================================================

class KlassBinaryMaccs(with_metaclass(ModelMetaClass, KlassSequential)):
    def __init__(self, args, log):
        super(KlassBinaryMaccs, self).__init__(args, log)

        # Define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = {"DEPENDENT": {"VARIABLE": "EC50_500", "SHAPE": [2], "TYPE": OSMModelData.CLASSES}
            , "INDEPENDENT": [{"VARIABLE": "MACCFP", "SHAPE": [167], "TYPE": OSMModelData.FLOAT64}]}

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "MACCS > BINARY CLASS (EC50) Classifier"

    def model_postfix(self):  # Must be unique for each model.
        return "bin_macc"

    def model_description(self):
        return ("A KERAS (TensorFlow) multi-layer Neural Network class classification model. \n"
                "This classifier analyzes the MACCS fingerprint against binary classes (EC50)")

    def model_define(self):  # Defines the modified sequential class with regularizers defined.

        model = Sequential()
        l2_param = 0.0
        l1_param = 0.0
        dropout_param = 0.0

        adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=5e-09)

        model.add(Dense(2048, input_dim=167, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param, l2_param), W_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(2048, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param, l2_param), W_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(512, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param, l2_param), W_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(64, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param, l2_param), W_constraint=maxnorm(3)))

        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(2, activation="softmax", init="normal"))
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

        return model



# ===============================================================================
# Keras Pattern Classifier, fits Binary classes to the DRAGON data.
# ===============================================================================

class KlassBinaryDragon(with_metaclass(ModelMetaClass, KlassSequential)):
    def __init__(self, args, log):
        super(KlassBinaryDragon, self).__init__(args, log)

        # Define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = {"DEPENDENT": {"VARIABLE": "EC50_500", "SHAPE": [2], "TYPE": OSMModelData.CLASSES}
            , "INDEPENDENT": [{"VARIABLE": "DRAGON", "SHAPE": [1666], "TYPE": OSMModelData.FLOAT64}]}

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "DRAGON > Any Binary Class (EC50) Classifier"

    def model_postfix(self):  # Must be unique for each model.
        return "bin_d"

    def model_description(self):
        return ("A KERAS (TensorFlow) multi-layer Neural Network class classification model. \n"
                "This classifier analyzes the DRAGON data against any binary class")

    def model_define(self):  # Defines the modified sequential class with regularizers defined.

        model = Sequential()
        l2_param = 0.0
        l1_param = 0.0
        dropout_param = 0.3

        adam = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=2e-09)

        model.add(Dense(2048, input_dim=1666, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param, l2_param), W_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(2048, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param,l2_param), W_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(2048, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param,l2_param), W_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(512, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param, l2_param), W_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(64, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param, l2_param), W_constraint=maxnorm(3)))

        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(2, activation = "softmax", init="normal"))
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

        return model

# ===============================================================================
# Keras Pattern Classifier, fits ION ACTIVITY to the DRAGON data.
# ===============================================================================

class KlassIonDragon(with_metaclass(ModelMetaClass, KlassSequential)):
    def __init__(self, args, log):
        super(KlassIonDragon, self).__init__(args, log)

        # Define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = {"DEPENDENT": {"VARIABLE": "ION_ACTIVITY", "SHAPE": [3], "TYPE": OSMModelData.CLASSES}
            , "INDEPENDENT": [{"VARIABLE": "DRAGON", "SHAPE": [1666], "TYPE": OSMModelData.FLOAT64}]}

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "DRAGON > ION_ACTIVITY Classifier"

    def model_postfix(self):  # Must be unique for each model.
        return "ion_d"

    def model_description(self):
        return ("A KERAS (TensorFlow) multi-layer Neural Network class classification model. \n"
                "This classifier analyzes the DRAGON data against ION_ACTIVITY")

    def model_define(self):  # Defines the modified sequential class with regularizers defined.

        model = Sequential()
        l2_param = 0.0
        l1_param = 0.0
        dropout_param = 0.0

        adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=5e-09)

        model.add(Dense(2048, input_dim=1666, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param, l2_param), W_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(2048, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param, l2_param), W_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(512, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param, l2_param), W_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(64, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param, l2_param), W_constraint=maxnorm(3)))

        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(3, activation="softmax", init="normal"))
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

        return model

# ===============================================================================
# Keras Pattern Classifier, fits Binary Classes to MORGAN2048 fingerprints
# ===============================================================================

class KlassBinaryMorgan(with_metaclass(ModelMetaClass, KlassSequential)):
    def __init__(self, args, log):
        super(KlassBinaryMorgan, self).__init__(args, log)

        # Define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = {"DEPENDENT": {"VARIABLE": "EC50_500", "SHAPE": [2], "TYPE": OSMModelData.CLASSES}
            , "INDEPENDENT": [{"VARIABLE": "MORGAN2048_4", "SHAPE": [2048], "TYPE": OSMModelData.FLOAT64}]}

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "MORGAN > Binary Class (EC50) Classifier"

    def model_postfix(self):  # Must be unique for each model.
        return "bin_m"

    def model_description(self):
        return ("A KERAS (TensorFlow) multi-layer Neural Network class classification model. \n"
                "This classifier analyzes MORGAN2048_n, TOPOLOGICAL2048 against any binary class")

    def model_define(self):  # Defines the modified sequential class with regularizers defined.


        model = Sequential()
        l2_param = 0.0
        l1_param = 0.0
        dropout_param = 0.0

        adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=5e-09)

        model.add(Dense(2048, input_dim=2048, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param, l2_param), W_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(2048, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param, l2_param), W_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(512, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param, l2_param), W_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(64, init="uniform", activation="relu"
                        , activity_regularizer=l1l2(l1_param, l2_param), W_constraint=maxnorm(3)))

        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(Dense(2, activation="softmax", init="normal"))
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

        return model

