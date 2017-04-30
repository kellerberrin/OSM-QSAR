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
from keras.regularizers import l2, l1_l2
from keras.layers.noise import GaussianDropout, GaussianNoise
from keras.models import load_model
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam, Adagrad, Adadelta
from keras.utils import np_utils

#from keras.utils.visualize_util import plot
import keras.backend as backend

from OSMBase import ModelMetaClass  # The virtual model class.
from OSMKerasBase import KerasClassifier
from OSMKerasDragon import KlassBinaryDragon, KlassIonDragon, TruncIonDragon
from OSMKerasFingerprint import  KlassIonMaccs, KlassIonMorgan, KlassBinaryMorgan
from OSMKerasCoulomb import CoulombMatrix
from OSMModelData import OSMModelData
from OSMSKLearnClassify import OSMSKLearnLOGC, OSMSKLearnNBC  # All The SKLearn Classifiers for the meta NN

# ================================================================================================
# A meta pattern classifier, everything and the kitchen sink, used for model development.
# ================================================================================================

class MetaSequential(with_metaclass(ModelMetaClass, KerasClassifier)):
    def __init__(self, args, log):
        super(MetaSequential, self).__init__(args, log)

        # Define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = {"DEPENDENT": {"VARIABLE": "ION_ACTIVE", "SHAPE": [2], "TYPE": OSMModelData.CLASSES}
            , "INDEPENDENT": [{"VARIABLE": "DRAGON", "SHAPE": [1666], "TYPE": OSMModelData.FLOAT64},
                              {"VARIABLE": "MORGAN2048_4", "SHAPE": [2048], "TYPE": OSMModelData.FLOAT64},
                              {"VARIABLE": "MORGAN2048_5", "SHAPE": [2048], "TYPE": OSMModelData.FLOAT64},
                              {"VARIABLE": "TOPOLOGICAL2048", "SHAPE": [2048], "TYPE": OSMModelData.FLOAT64},
                              {"VARIABLE": "MORGAN2048_1", "SHAPE": [2048], "TYPE": OSMModelData.FLOAT64},
                              {"VARIABLE": "MACCFP", "SHAPE": [167], "TYPE": OSMModelData.FLOAT64},
                              {"VARIABLE": "TRUNC_DRAGON", "SHAPE": [100], "TYPE": OSMModelData.FLOAT64},
                              {"VARIABLE": "COULOMB_ARRAY", "SHAPE": [2916], "TYPE": OSMModelData.FLOAT64}]}


        self.model_define_meta(args, log)

    def model_define_meta(self, args, log):


        ion_d_args = copy.deepcopy(args) #ensure that args cannot be side-swiped.
        ion_d_args.indepList = ["DRAGON"]
        ion_d_args.dependVar = "ION_ACTIVITY"
        ion_d_args.train = 0
        ion_d_args.epoch = 200
        ion_d_args.loadFilename = os.path.join(ion_d_args.postfixDirectory, "DRAGON")
        self.dnn_dragon = KlassIonDragon(ion_d_args, log)

        ion_td_args = copy.deepcopy(args) #ensure that args cannot be side-swiped.
        ion_td_args.indepList = ["TRUNC_DRAGON"]
        ion_td_args.dependVar = "ION_ACTIVITY"
        ion_td_args.train = 0
        ion_td_args.epoch = 160
        ion_td_args.loadFilename = os.path.join(ion_td_args.postfixDirectory, "TRUNC100")
        self.dnn_trunc_dragon = TruncIonDragon(ion_td_args, log)

        ion_c_args = copy.deepcopy(args) #ensure that args cannot be side-swiped.
        ion_c_args.indepList = ["COULOMB_ARRAY"]
        ion_c_args.dependVar = "ION_ACTIVITY"
        ion_c_args.train = 0
        ion_c_args.epoch = 100
        ion_c_args.loadFilename = os.path.join(ion_c_args.postfixDirectory, "COULOMB")
        self.dnn_coulomb = CoulombMatrix(ion_c_args, log)


        binion_d_args = copy.deepcopy(args) #ensure that args cannot be side-swiped.
        binion_d_args.indepList = ["DRAGON"]
        binion_d_args.dependVar = "ION_ACTIVE"
        binion_d_args.train = 0
        binion_d_args.epoch = 140
        binion_d_args.loadFilename = os.path.join(ion_d_args.postfixDirectory, "BIN_DRAGON")
        self.dnn_bin_dragon = KlassBinaryDragon(binion_d_args, log)

        ion_m1_args = copy.deepcopy(args) #ensure that args cannot be side-swiped.
        ion_m1_args.indepList = ["MORGAN2048_1"]
        ion_m1_args.dependVar = "ION_ACTIVITY"
        ion_m1_args.train = 0
        ion_m1_args.epoch = 300
        ion_m1_args.loadFilename = os.path.join(ion_d_args.postfixDirectory, "ION_MORGAN1")
        self.dnn_morgan1 = KlassIonMorgan(ion_m1_args, log)

        ion_m5_args = copy.deepcopy(args) #ensure that args cannot be side-swiped.
        ion_m5_args.indepList = ["MORGAN2048_5"]
        ion_m5_args.dependVar = "ION_ACTIVITY"
        ion_m5_args.train = 0
        ion_m5_args.epoch = 120
        ion_m5_args.loadFilename = os.path.join(ion_d_args.postfixDirectory, "ION_MORGAN5")
        self.dnn_morgan5 = KlassIonMorgan(ion_m5_args, log)

        ec50_m5_args = copy.deepcopy(args) #ensure that args cannot be side-swiped.
        ec50_m5_args.indepList = ["MORGAN2048_5"]
        ec50_m5_args.dependVar = "EC50_500"
        ec50_m5_args.train = 0
        ec50_m5_args.epoch = 540
        ec50_m5_args.loadFilename = os.path.join(ion_d_args.postfixDirectory, "EC50_500_MORGAN5")
        self.dnn_ec50_m5 = KlassBinaryMorgan(ec50_m5_args, log)

        ion_top_args = copy.deepcopy(args) #ensure that args cannot be side-swiped.
        ion_top_args.indepList = ["TOPOLOGICAL2048"]
        ion_top_args.dependVar = "ION_ACTIVITY"
        ion_top_args.train = 0
        ion_top_args.epoch = 480
        ion_top_args.loadFilename = os.path.join(ion_d_args.postfixDirectory, "ION_TOPOLOGICAL")
        self.dnn_top = KlassIonMorgan(ion_top_args, log)

        ion_macc_args = copy.deepcopy(args) #ensure that args cannot be side-swiped.
        ion_macc_args.indepList = ["MACCFP"]
        ion_macc_args.dependVar = "ION_ACTIVITY"
        ion_macc_args.train = 0
        ion_macc_args.epoch = 420
        ion_macc_args.loadFilename = os.path.join(ion_d_args.postfixDirectory, "ION_MACCS")
        self.dnn_macc = KlassIonMaccs(ion_macc_args, log)

        logc_args = copy.deepcopy(args)
        logc_args.indepList = ["MORGAN2048_5"]
        logc_args.dependVar = "EC50_500"
        self.logc = OSMSKLearnLOGC(logc_args, log)

        nbc_args = copy.deepcopy(args)
        nbc_args.indepList = ["MORGAN2048_4"]
        nbc_args.dependVar = "ION_ACTIVITY"
        self.nbc = OSMSKLearnNBC(nbc_args, log)

    def model_meta_train(self):
        self.nbc.initialize(self.raw_data)
        self.logc.initialize(self.raw_data)
        self.dnn_dragon.initialize(self.raw_data)
        self.dnn_trunc_dragon.initialize(self.raw_data)
        self.dnn_coulomb.initialize(self.raw_data)
        self.dnn_morgan1.initialize(self.raw_data)
        self.dnn_morgan5.initialize(self.raw_data)
        self.dnn_top.initialize(self.raw_data)
        self.dnn_macc.initialize(self.raw_data)
        self.dnn_ec50_m5.initialize(self.raw_data)
        self.dnn_bin_dragon.initialize(self.raw_data)


    def model_name(self):
        return "Meta DNN Classifier"

    def model_postfix(self):  # Must be unique for each model.
        return "ion_meta"

    def model_description(self):
        return ("A multi-layer Neural Network that uses other classification model probability functions as input. \n"
                "The other classification models are pre-trained")

    def model_define(self):  # Defines the modified sequential class with regularizers defined.

        self.model_meta_train()
        return self.model_arch()

    def model_arch(self):  # Defines the modified sequential class with regularizers defined.

        model = Sequential()
        dropout_param = 0.0

        adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=5e-09)

        model.add(Dense(16, input_dim=2, kernel_initializer="uniform", activation="relu", kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(Dense(32, kernel_initializer="uniform", activation="relu", kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(Dense(32, init="uniform", activation="relu", kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(Dense(16, kernel_initializer="uniform", activation="relu", kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(Dense(2, activation = "softmax", kernel_initializer="normal"))
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

        return model

    def model_prediction(self, data):

        predictions = self.model.predict_classes(self.input_probability(data), verbose=0)
        classes = self.model_enumerate_classes()
        class_list = []
        for predict in predictions:
            class_list.append(classes[predict])
        return {"prediction": class_list, "actual": data.target_data()}

    def model_evaluate(self, data):
        classes = self.model_enumerate_classes()
        class_list = data.target_data()
        index_list = []
        for a_class in class_list:
            index_list.append(classes.index(a_class))
        binary_labels = np_utils.to_categorical(index_list)
        score = self.model.evaluate(self.input_probability(data), binary_labels, verbose=0)
        return score

    def model_probability(self, data):  # probabilities are returned as a numpy.shape = (samples, classes)
        prob =self.model.predict_proba(self.input_probability(data))
        prob_list = list(prob)
        return {"probability": prob_list}

    def train_epoch(self, epoch):

        classes = self.model_enumerate_classes()
        class_list = self.data.training().target_data()
        index_list = []
        for a_class in class_list:
            index_list.append(classes.index(a_class))
        binary_labels = np_utils.to_categorical(index_list)

        hist = self.model.fit(self.input_probability(self.data.training()), binary_labels,
                       validation_split=self.args.holdOut, epochs=epoch, verbose=1)

        self.train_history("model_aux.csv", hist.history, epoch)

    def epoch_read(self, epoch):
        self.model_meta_train()
        file_name = self.args.loadFilename + "_" + "{}".format(epoch) + ".krs"
        self.log.info("KERAS - Loading Trained %s Model in File: %s", self.model_name(), file_name)
        model = load_model(file_name)
        return model

    def input_probability(self, data):
        logc_prob = self.logc.model.predict_proba(data.input_data()["MORGAN2048_5"])
        nbc_prob = self.nbc.model.predict_proba(data.input_data()["MORGAN2048_4"])
        dnn_dragon_prob = self.dnn_dragon.model.predict_proba(data.input_data()["DRAGON"])
        dnn_dragon_prob = np.asarray(dnn_dragon_prob)
        dnn_trunc_dragon_prob = self.dnn_trunc_dragon.model.predict_proba(data.input_data()["TRUNC_DRAGON"])
        dnn_trunc_dragon_prob = np.asarray(dnn_trunc_dragon_prob)
        dnn_coulomb_prob = self.dnn_coulomb.model.predict_proba(data.input_data()["COULOMB_ARRAY"])
        dnn_coulomb_prob = np.asarray(dnn_coulomb_prob)
        dnn_m1_prob = self.dnn_morgan1.model.predict_proba(data.input_data()["MORGAN2048_1"])
        dnn_m1_prob = np.asarray(dnn_m1_prob)
        dnn_m5_prob = self.dnn_morgan5.model.predict_proba(data.input_data()["MORGAN2048_5"])
        dnn_m5_prob = np.asarray(dnn_m5_prob)
        dnn_top_prob = self.dnn_top.model.predict_proba(data.input_data()["TOPOLOGICAL2048"])
        dnn_top_prob = np.asarray(dnn_top_prob)
        dnn_macc_prob = self.dnn_macc.model.predict_proba(data.input_data()["MACCFP"])
        dnn_macc_prob = np.asarray(dnn_macc_prob)
        dnn_ec50_prob = self.dnn_ec50_m5.model.predict_proba(data.input_data()["MORGAN2048_5"])
        dnn_ec50_prob = np.asarray(dnn_ec50_prob)
        dnn_bin_dragon_prob = self.dnn_bin_dragon.model.predict_proba(data.input_data()["DRAGON"])
        dnn_bin_dragon_prob = np.asarray(dnn_bin_dragon_prob)

#        print("dragon", dnn_dragon_prob.shape, "ec50", dnn_ec50_prob.shape)
#        print("dragon", dnn_dragon_prob[:,0].shape, "ec50", dnn_ec50_prob[:,0].shape)
#        prob = np.column_stack((dnn_dragon_prob[:,0], logc_prob[:,0]))
        prob = np.column_stack((dnn_trunc_dragon_prob[:,0], logc_prob[:,0]))

#        print("prob", prob.shape)

#        prob = np.concatenate((dnn_dragon_prob[:,0], dnn_ec50_prob[:,0]),axis=1)

        #        prob = np.concatenate((dnn_macc_prob, dnn_dragon_prob, dnn_m1_prob, dnn_m5_prob, dnn_top_prob),axis=1)
        return prob

# ================================================================================================
# The Final prediction model used in the OSM competition.
# ================================================================================================

class OSMMeta(with_metaclass(ModelMetaClass, KerasClassifier)):
    def __init__(self, args, log):
        super(OSMMeta, self).__init__(args, log)

        # Define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = {"DEPENDENT": {"VARIABLE": "ION_ACTIVE", "SHAPE": [2], "TYPE": OSMModelData.CLASSES}
            , "INDEPENDENT": [{"VARIABLE": "DRAGON", "SHAPE": [1666], "TYPE": OSMModelData.FLOAT64},
                              {"VARIABLE": "MORGAN2048_5", "SHAPE": [2048], "TYPE": OSMModelData.FLOAT64} ]}

        self.model_define_meta(args, log)

    def model_define_meta(self, args, log):

        ion_d_args = copy.deepcopy(args)  # ensure that args cannot be side-swiped.
        ion_d_args.indepList = ["DRAGON"]
        ion_d_args.dependVar = "ION_ACTIVITY"
        ion_d_args.train = 0
        ion_d_args.epoch = 625
        ion_d_args.loadFilename = os.path.join(ion_d_args.postfixDirectory, "ION_DRAGON")
        self.dnn_dragon = KlassIonDragon(ion_d_args, log)

        logc_args = copy.deepcopy(args)
        logc_args.indepList = ["MORGAN2048_5"]
        logc_args.dependVar = "EC50_500"
        self.logc = OSMSKLearnLOGC(logc_args, log)

    def model_meta_train(self):
        self.logc.initialize(self.raw_data)
        self.dnn_dragon.initialize(self.raw_data)

    def model_name(self):
        return "The OSM Competition Classifier"

    def model_postfix(self):  # Must be unique for each model.
        return "osm"

    def model_description(self):
        return (
        "A meta multi-layer Neural Network that uses other classification model probability functions as input. \n"
        "This classifier model was developed for the OSM molecular ION Activity Competition.\n"
        "It takes the input of an optimal NN that uses the DRAGON data to examine and classify molecular structure\n"
        "and an SKLearn logistic classifier that estimates the molecular potency EC50 potency at 500nMol.\n"
        "The probability maps of these classifiers and then optimally combined in a multi-layer NN to estimate\n"
        "molecular ion activity.")

    def model_define(self):  # Defines the modified sequential class with regularizers defined.

        self.model_meta_train()
        return self.model_arch()

    def model_arch(self):  # Defines the modified sequential class with regularizers defined.

        model = Sequential()
        dropout_param = 0.0

        adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=5e-09)

        model.add(Dense(16, input_dim=2, kernel_initializer="uniform", activation="relu", kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(Dense(32, kernel_initializer="uniform", activation="relu", kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(Dense(32, kernel_initializer="uniform", activation="relu", kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(Dense(16, kernel_initializer="uniform", activation="relu", kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(Dense(2, activation="softmax", kernel_initializer="normal"))
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

        return model

    def model_prediction(self, data):

        predictions = self.model.predict_classes(self.input_probability(data), verbose=0)
        classes = self.model_enumerate_classes()
        class_list = []
        for predict in predictions:
            class_list.append(classes[predict])
        return {"prediction": class_list, "actual": data.target_data()}

    def model_evaluate(self, data):
        classes = self.model_enumerate_classes()
        class_list = data.target_data()
        index_list = []
        for a_class in class_list:
            index_list.append(classes.index(a_class))
        binary_labels = np_utils.to_categorical(index_list)
        score = self.model.evaluate(self.input_probability(data), binary_labels, verbose=0)
        return score

    def model_probability(self, data):  # probabilities are returned as a numpy.shape = (samples, classes)
        prob = self.model.predict_proba(self.input_probability(data))
        prob_list = list(prob)
        return {"probability": prob_list}

    def train_epoch(self, epoch):

        classes = self.model_enumerate_classes()
        class_list = self.data.training().target_data()
        index_list = []
        for a_class in class_list:
            index_list.append(classes.index(a_class))
        binary_labels = np_utils.to_categorical(index_list)

        hist = self.model.fit(self.input_probability(self.data.training()), binary_labels,
                              validation_split=self.args.holdOut, epochs=epoch, verbose=1)

        self.train_history("model_aux.csv", hist.history, epoch)

    def epoch_read(self, epoch):
        self.model_meta_train()
        file_name = self.args.loadFilename + "_" + "{}".format(epoch) + ".krs"
        self.log.info("KERAS - Loading Trained %s Model in File: %s", self.model_name(), file_name)
        model = load_model(file_name)
        return model

    def input_probability(self, data):

        logc_prob = self.logc.model.predict_proba(data.input_data()["MORGAN2048_5"])

        dnn_dragon_prob = self.dnn_dragon.model.predict_proba(data.input_data()["DRAGON"])
        dnn_dragon_prob = np.asarray(dnn_dragon_prob)

        prob = np.column_stack((dnn_dragon_prob[:, 0], logc_prob[:, 0]))

        return prob

# ================================================================================================
# The Final prediction model used in the OSM competition.
# ================================================================================================

class OSMTruncMeta(with_metaclass(ModelMetaClass, KerasClassifier)):
    def __init__(self, args, log):
        super(OSMTruncMeta, self).__init__(args, log)

        # Define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = {"DEPENDENT": {"VARIABLE": "ION_ACTIVE", "SHAPE": [2], "TYPE": OSMModelData.CLASSES}
            , "INDEPENDENT": [{"VARIABLE": "TRUNC_DRAGON", "SHAPE": [100], "TYPE": OSMModelData.FLOAT64},
                              {"VARIABLE": "MORGAN2048_5", "SHAPE": [2048], "TYPE": OSMModelData.FLOAT64} ]}

        self.model_define_meta(args, log)

    def model_define_meta(self, args, log):

        ion_d_args = copy.deepcopy(args)  # ensure that args cannot be side-swiped.
        ion_d_args.indepList = ["TRUNC_DRAGON"]
        ion_d_args.dependVar = "ION_ACTIVITY"
        ion_d_args.train = 0
        ion_d_args.epoch = 160
        ion_d_args.loadFilename = os.path.join(ion_d_args.postfixDirectory, "TRUNC100")
        self.dnn_dragon = TruncIonDragon(ion_d_args, log)

        logc_args = copy.deepcopy(args)
        logc_args.indepList = ["MORGAN2048_5"]
        logc_args.dependVar = "EC50_500"
        self.logc = OSMSKLearnLOGC(logc_args, log)

    def model_meta_train(self):
        self.logc.initialize(self.raw_data)
        self.dnn_dragon.initialize(self.raw_data)

    def model_name(self):
        return "The Truncated Dragon Meta Classifier"

    def model_postfix(self):  # Must be unique for each model.
        return "td_meta"

    def model_description(self):
        return (
        "A meta multi-layer Neural Network that uses other classification model probability functions as input. \n"
        "This classifier model was developed for the OSM molecular ION Activity Competition.\n"
        "It takes the input of an optimal NN that uses truncated DRAGON data (100 fields)\n"
        "and an SKLearn logistic classifier that estimates the molecular potency EC50 potency at 500nMol.\n"
        "The probability maps of these classifiers and then optimally combined in a multi-layer NN to estimate\n"
        "molecular ion activity.")

    def model_define(self):  # Defines the modified sequential class with regularizers defined.

        self.model_meta_train()
        return self.model_arch()

    def model_arch(self):  # Defines the modified sequential class with regularizers defined.

        model = Sequential()
        dropout_param = 0.5
        activation_fn = "relu"
        Gaussian_noise = 1

        adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=5e-09)

        model.add(Dense(32, input_dim=2, kernel_initializer="uniform", activation=activation_fn, kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(GaussianNoise(Gaussian_noise))

        model.add(Dense(128, kernel_initializer="uniform", activation=activation_fn, kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(GaussianNoise(Gaussian_noise))

        model.add(Dense(32, kernel_initializer="uniform", activation=activation_fn, kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(BatchNormalization())
        model.add(GaussianNoise(Gaussian_noise))

        model.add(Dense(2, activation="softmax", kernel_initializer="normal"))
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

        return model

    def model_prediction(self, data):

        predictions = self.model.predict_classes(self.input_probability(data), verbose=0)
        classes = self.model_enumerate_classes()
        class_list = []
        for predict in predictions:
            class_list.append(classes[predict])
        return {"prediction": class_list, "actual": data.target_data()}

    def model_evaluate(self, data):
        classes = self.model_enumerate_classes()
        class_list = data.target_data()
        index_list = []
        for a_class in class_list:
            index_list.append(classes.index(a_class))
        binary_labels = np_utils.to_categorical(index_list)
        score = self.model.evaluate(self.input_probability(data), binary_labels, verbose=0)
        return score

    def model_probability(self, data):  # probabilities are returned as a numpy.shape = (samples, classes)
        prob = self.model.predict_proba(self.input_probability(data))
        prob_list = list(prob)
        return {"probability": prob_list}

    def train_epoch(self, epoch):

        classes = self.model_enumerate_classes()
        class_list = self.data.training().target_data()
        index_list = []
        for a_class in class_list:
            index_list.append(classes.index(a_class))
        binary_labels = np_utils.to_categorical(index_list)

        hist = self.model.fit(self.input_probability(self.data.training()), binary_labels,
                              validation_split=self.args.holdOut, epochs=epoch, verbose=1)

        self.train_history("model_aux.csv", hist.history, epoch)

    def epoch_read(self, epoch):
        self.model_meta_train()
        file_name = self.args.loadFilename + "_" + "{}".format(epoch) + ".krs"
        self.log.info("KERAS - Loading Trained %s Model in File: %s", self.model_name(), file_name)
        model = load_model(file_name)
        return model

    def input_probability(self, data):

        logc_prob = self.logc.model.predict_proba(data.input_data()["MORGAN2048_5"])

        dnn_dragon_prob = self.dnn_dragon.model.predict_proba(data.input_data()["TRUNC_DRAGON"])
        dnn_dragon_prob = np.asarray(dnn_dragon_prob)

        prob = np.column_stack((dnn_dragon_prob[:, 0], logc_prob[:, 0]))

        return prob





