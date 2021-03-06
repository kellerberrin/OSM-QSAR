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
from keras.layers.noise import GaussianDropout, GaussianNoise
from keras.layers import normalization, BatchNormalization
from keras.regularizers import l2, l1_l2
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam, Adagrad, Adadelta
from keras.utils import np_utils

#from keras.utils.visualize_util import plot
import keras.backend as backend

from OSMBase import ModelMetaClass  # The virtual model class.
from OSMKerasBase import KlassSequential
from OSMModelData import OSMModelData
from OSMAnalysis import OSMSensitivity, OSMDragonSensitivity



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
        dropout_param = 0.5
        Gaussian_noise = 1
        initializer = "uniform"
        activation_func = "relu"

        adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=5e-09)

        model.add(Dense(2048, input_dim=1666, kernel_initializer=initializer, activation=activation_func
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

        model.add(Dense(2, activation="softmax", kernel_initializer=initializer))
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

    def model_analytics(self, data):

        self.log.info("Calculating Neural Network DRAGON field sensitivity, may take a few moments....")

        def analysis_probability(x):
            return self.model.predict_proba(x, verbose=0)

        func = analysis_probability

        Sens = OSMDragonSensitivity(self.args, self.log)

        result_dict = {}

        if self.args.extendFlag and False:

            self.log.info("Analytics - Calculating Field Step (1Q) Sensitivity")
            dragon_1q_sensitivity = Sens.calc_dragon_step_sensitivity(func, data.input_data(), 10, 25,
                                                                    self.raw_data.get_dragon_fields())
            result_dict = {"1Q_SENSITIVITY": dragon_1q_sensitivity }

        if self.args.extendFlag:

            self.log.info("Analytics - Calculating Field Step (Median) Sensitivity")
            dragon_median_sensitivity = Sens.calc_dragon_step_sensitivity(func, data.input_data(), 10, 50,
                                                                    self.raw_data.get_dragon_fields())
            result_dict["MEDIAN_SENSITIVITY"] = dragon_median_sensitivity

        if self.args.extendFlag and False:

            self.log.info("Analytics - Calculating Field Step (3Q) Sensitivity")
            dragon_3q_sensitivity = Sens.calc_dragon_step_sensitivity(func, data.input_data(), 10, 75,
                                                                    self.raw_data.get_dragon_fields())
            result_dict["3Q_SENSITIVITY"] = dragon_3q_sensitivity

        if self.args.extendFlag:

            self.log.info("Analytics - Calculating Field Derivatives")
            dragon_derivative = Sens.calc_dragon_derivative(func, data.input_data(), 0.01,
                                                       self.raw_data.get_dragon_fields())
            result_dict["DERIVATIVE"] = dragon_derivative

        if self.args.extendFlag and False:  # too time consuming.
            self.log.info("Analytics - Calculating Field Partial Derivatives")
            dragon_partial = Sens.calc_dragon_partial(func, data.input_data(), 0.01,
                                                            self.raw_data.get_dragon_fields())
            result_dict["PARTIAL"] = dragon_partial

        return result_dict

    def model_define(self):  # Defines the modified sequential class with regularizers defined.

        model = Sequential()
        l2_param = 0.0
        l1_param = 0.0
        dropout_param = 0.5
        Gaussian_noise = 1
        initializer = "uniform"
        activation_func = "relu"

        adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=5e-09)

        model.add(Dense(2048, input_dim=1666, kernel_initializer=initializer, activation=activation_func
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
# Keras Pattern Classifier, fits ION ACTIVITY to the DRAGON data.
# ===============================================================================

class TruncIonDragon(with_metaclass(ModelMetaClass, KlassSequential)):
    def __init__(self, args, log):
        super(TruncIonDragon, self).__init__(args, log)

        # Define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = {"DEPENDENT": {"VARIABLE": "ION_ACTIVITY", "SHAPE": [3], "TYPE": OSMModelData.CLASSES}
            , "INDEPENDENT": [{"VARIABLE": "TRUNC_DRAGON", "SHAPE": [100], "TYPE": OSMModelData.FLOAT64}]}

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "TRUNC_DRAGON > ION_ACTIVITY Classifier"

    def model_postfix(self):  # Must be unique for each model.
        return "ion_td"

    def model_description(self):
        return ("A KERAS (TensorFlow) multi-layer Neural Network class classification model. \n"
                "This classifier analyzes the TRUNC_DRAGON data against ION_ACTIVITY")

    def model_analytics(self, data):

        self.log.info("Calculating Neural Network TRUNC_DRAGON field sensitivity, may take a few moments....")
        func = lambda  x: self.model.predict_proba(x, verbose=0)
        Sens = OSMDragonSensitivity(self.args, self.log)

        result_dict = {}

        if self.args.extendFlag and False:

            self.log.info("Analytics - Calculating Field Step (1Q) Sensitivity")
            dragon_1q_sensitivity = Sens.calc_dragon_step_sensitivity(func, data.input_data(), 10, 25,
                                                                    self.raw_data.get_truncated_dragon_fields())
            result_dict["1Q_SENSITIVITY"] = dragon_1q_sensitivity


        if self.args.extendFlag:

            self.log.info("Analytics - Calculating Field Step (Median) Sensitivity")
            dragon_median_sensitivity = Sens.calc_dragon_step_sensitivity(func, data.input_data(), 10, 50,
                                                                    self.raw_data.get_truncated_dragon_fields())
            result_dict["MEDIAN_SENSITIVITY"] = dragon_median_sensitivity


        if self.args.extendFlag and False:

            self.log.info("Analytics - Calculating Field Step (3Q) Sensitivity")
            dragon_3q_sensitivity = Sens.calc_dragon_step_sensitivity(func, data.input_data(), 10, 75,
                                                                    self.raw_data.get_truncated_dragon_fields())
            result_dict["3Q_SENSITIVITY"] =  dragon_3q_sensitivity

        if self.args.extendFlag:

            self.log.info("Analytics - Calculating Field Derivatives")
            dragon_derivative = Sens.calc_dragon_derivative(func, data.input_data(), 0.01,
                                                            self.raw_data.get_truncated_dragon_fields())
            result_dict["DERIVATIVE"] =  dragon_derivative

        if self.args.extendFlag and False:

            self.log.info("Analytics - Calculating Field Partial Derivatives")
            dragon_partial = Sens.calc_dragon_partial(func, data.input_data(), 0.01,
                                                    self.raw_data.get_truncated_dragon_fields())
            result_dict["PARTIAL"] = dragon_partial

        return result_dict

    def model_define(self):  # Defines the modified sequential class with regularizers defined.

        model = Sequential()
        l2_param = 0.0
        l1_param = 0.0
        dropout_param = 0.5
        Gaussian_noise = 1
        initializer = "uniform"
        activation_func = "tanh"

        adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=5e-09)

        model.add(Dense(1024, input_dim=100, kernel_initializer=initializer, activation=activation_func
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

        model.add(Dense(512, kernel_initializer=initializer, activation=activation_func
                        , activity_regularizer=l1_l2(l1_param, l2_param), kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(GaussianNoise(Gaussian_noise))

        model.add(Dense(64, kernel_initializer=initializer, activation=activation_func
                        , activity_regularizer=l1_l2(l1_param, l2_param), kernel_constraint=maxnorm(3)))

        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(GaussianNoise(Gaussian_noise))

        model.add(Dense(3, activation="softmax", kernel_initializer=initializer))
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

        return model

# ===============================================================================
# Keras Pattern Classifier, fits ION ACTIVITY to the DRAGON data.
# ===============================================================================

class TruncBinDragon(with_metaclass(ModelMetaClass, KlassSequential)):
    def __init__(self, args, log):
        super(TruncBinDragon, self).__init__(args, log)

        # Define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = {"DEPENDENT": {"VARIABLE": "ION_ACTIVE", "SHAPE": [2], "TYPE": OSMModelData.CLASSES}
            , "INDEPENDENT": [{"VARIABLE": "TRUNC_DRAGON", "SHAPE": [100], "TYPE": OSMModelData.FLOAT64}]}

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "TRUNC_DRAGON > BINARY ION_ACTIVE Classifier"

    def model_postfix(self):  # Must be unique for each model.
        return "bin_td"

    def model_description(self):
        return ("A KERAS (TensorFlow) multi-layer Neural Network class classification model. \n"
                "This classifier analyzes the TRUNC_DRAGON data against BINARY ION_ACTIVE")

    def model_analytics(self, data):

        self.log.info("Calculating Neural Network TRUNC_DRAGON field sensitivity, may take a few moments....")
        func = lambda  x: self.model.predict_proba(x, verbose=0)
        Sens = OSMDragonSensitivity(self.args, self.log)

        result_dict = {}

        if self.args.extendFlag and False:

            self.log.info("Analytics - Calculating Field Step (1Q) Sensitivity")
            dragon_1q_sensitivity = Sens.calc_dragon_step_sensitivity(func, data.input_data(), 10, 25,
                                                                    self.raw_data.get_truncated_dragon_fields())
            result_dict["1Q_SENSITIVITY"] = dragon_1q_sensitivity


        if self.args.extendFlag:

            self.log.info("Analytics - Calculating Field Step (Median) Sensitivity")
            dragon_median_sensitivity = Sens.calc_dragon_step_sensitivity(func, data.input_data(), 10, 50,
                                                                    self.raw_data.get_truncated_dragon_fields())
            result_dict["MEDIAN_SENSITIVITY"] = dragon_median_sensitivity


        if self.args.extendFlag and False:

            self.log.info("Analytics - Calculating Field Step (3Q) Sensitivity")
            dragon_3q_sensitivity = Sens.calc_dragon_step_sensitivity(func, data.input_data(), 10, 75,
                                                                    self.raw_data.get_truncated_dragon_fields())
            result_dict["3Q_SENSITIVITY"] =  dragon_3q_sensitivity

        if self.args.extendFlag:

            self.log.info("Analytics - Calculating Field Derivatives")
            dragon_derivative = Sens.calc_dragon_derivative(func, data.input_data(), 0.01,
                                                            self.raw_data.get_truncated_dragon_fields())
            result_dict["DERIVATIVE"] =  dragon_derivative

        if self.args.extendFlag and False:

            self.log.info("Analytics - Calculating Field Partial Derivatives")
            dragon_partial = Sens.calc_dragon_partial(func, data.input_data(), 0.01,
                                                    self.raw_data.get_truncated_dragon_fields())
            result_dict["PARTIAL"] = dragon_partial

        return result_dict

    def model_define(self):  # Defines the modified sequential class with regularizers defined.

        model = Sequential()
        l2_param = 0.0
        l1_param = 0.0
        dropout_param = 0.5
        Gaussian_noise = 1
        initializer = "uniform"
        activation_func = "tanh"

        adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=5e-09)

        model.add(Dense(1024, input_dim=100, kernel_initializer=initializer, activation=activation_func
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

        model.add(Dense(512, kernel_initializer=initializer, activation=activation_func
                        , activity_regularizer=l1_l2(l1_param, l2_param), kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(GaussianNoise(Gaussian_noise))

        model.add(Dense(64, kernel_initializer=initializer, activation=activation_func
                        , activity_regularizer=l1_l2(l1_param, l2_param), kernel_constraint=maxnorm(3)))

        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(GaussianNoise(Gaussian_noise))

        model.add(Dense(2, activation="softmax", kernel_initializer=initializer))
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

        return model

