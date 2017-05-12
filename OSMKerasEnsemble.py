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
from OSMKerasCoulomb import CoulombMatrix, CoulombConvolution
from OSMModelData import OSMModelData
from OSMSKLearnClassify import OSMSKLearnLOGC, OSMSKLearnNBC  # All The SKLearn Classifiers for the meta NN

# ================================================================================================
# A meta pattern classifier, everything and the kitchen sink, used for model development.
# ================================================================================================

class EnsembleSequential(with_metaclass(ModelMetaClass, KerasClassifier)):
    def __init__(self, args, log):
        super(EnsembleSequential, self).__init__(args, log)

        # Define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = {"DEPENDENT": {"VARIABLE": "ION_ACTIVITY", "SHAPE": [3], "TYPE": OSMModelData.CLASSES}
            , "INDEPENDENT": [{"VARIABLE": "DRAGON", "SHAPE": [1666], "TYPE": OSMModelData.FLOAT64} ] }

        self.ensembles = self.model_define_ensemble(args, log)

    def model_define_ensemble(self, args, log):

        ensemble_file_list = [ { "File": "Run1", "Epochs" : 400 },
                          {"File": "Run2", "Epochs": 260},
                          {"File": "Run3", "Epochs": 220},
                          {"File": "Run4", "Epochs": 220},
                          {"File": "Run5", "Epochs": 260},
                          {"File": "Run6", "Epochs": 280},
                          {"File": "Run7", "Epochs": 280},
                          {"File": "Run8", "Epochs": 340},
                          {"File": "Run9", "Epochs": 340},
                          {"File": "Run10", "Epochs": 200} ]



        ensembles = []

        for ensemble_file in ensemble_file_list:

            ensemble_args = copy.deepcopy(args)  # ensure that args cannot be side-swiped.
            ensemble_args.indepList = ["DRAGON"]
            ensemble_args.dependVar = "ION_ACTIVITY"
            ensemble_args.train = 0
            ensemble_args.epoch = ensemble_file["Epochs"]
            ensemble_args.loadFilename = os.path.join(ensemble_args.postfixDirectory, ensemble_file["File"])

            ensembles.append(KlassIonDragon(ensemble_args, log))

        return ensembles

    def model_ensemble_train(self):

        for ensemble in self.ensembles:
            ensemble.initialize(self.raw_data)


    def model_name(self):
        return "Ensemble DNN Classifier"

    def model_postfix(self):  # Must be unique for each model.
        return "ion_ens"

    def model_description(self):
        return ("A Neural Network that uses an ensemble of other classifiers as input. \n"
                "The other classification models are pre-trained")

    def model_define(self):  # Defines the modified sequential class with regularizers defined.

        self.model_ensemble_train()
        return self.model_arch()

    def model_arch(self):  # Defines the modified sequential class with regularizers defined.

        model = Sequential()
        dropout_param = 0.2
        activation = "relu"
        initializer = "uniform"

        adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=5e-09)

        model.add(Dense(8, input_dim=len(self.ensembles), kernel_initializer=initializer, activation=activation, kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(Dense(16,  kernel_initializer=initializer, activation=activation, kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(Dense(16,  kernel_initializer=initializer, activation=activation, kernel_constraint=maxnorm(3)))
        model.add(Dropout(dropout_param))
        model.add(Dense(3, activation = "softmax", kernel_initializer="normal"))
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
        self.model_ensemble_train()
        file_name = self.args.loadFilename + "_" + "{}".format(epoch) + ".krs"
        self.log.info("KERAS - Loading Trained %s Model in File: %s", self.model_name(), file_name)
        model = load_model(file_name)
        return model

    def input_probability(self, data):

        prob_list = []

        for ensemble in self.ensembles:

            prob = ensemble.model.predict_proba(data.input_data())
            prob = np.asarray(prob)
            prob_list.append(prob[:,0])

        prob_columns = np.column_stack(prob_list)

        return prob_columns

