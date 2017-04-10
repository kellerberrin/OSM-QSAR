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
from OSMClassify import OSMClassification
from OSMGraphics import OSMSimilarityMap
from OSMIterative import OSMIterative
from OSMGraphics import OSMDragonMap

# ===============================================================================
# Base class for the Keras neural network classifiers.
# ===============================================================================

class KerasClassifier(OSMClassification):

    def __init__(self, args, log):
        super(KerasClassifier, self).__init__(args, log)

        self.default_epochs = 1000

        self.iterative = OSMIterative(self)

    def model_write(self):
        self.iterative.write()

    def model_read(self):
        return self.iterative.read()

    def model_train(self):
        self.iterative.train(self.default_epochs)

    def epoch_write(self, epoch):
        file_name = self.args.saveFilename + "_" + "{}".format(epoch) + ".krs"
        self.log.info("KERAS - Saving Trained %s Model in File: %s", self.model_name(), file_name)
        self.model.save(file_name)

    def epoch_read(self, epoch):
        file_name = self.args.loadFilename + "_" + "{}".format(epoch) + ".krs"
        self.log.info("KERAS - Loading Trained %s Model in File: %s", self.model_name(), file_name)
        model = load_model(file_name)
        return model

    def model_epochs(self):
        return self.iterative.trained_epochs()


    def model_graphics(self):
        self.similarity_graphics()
        self.dragon_graphics()

    def train_history(self, file_name, history, epoch):

        model_file_path = os.path.join(self.args.postfixDirectory, file_name)
        total_epochs = self.model_epochs()
        begin_epoch = total_epochs - epoch + 1

        try:

            with open(model_file_path, 'a') as stats_file:

                for idx in range(epoch):

                    if self.args.holdOut > 0.0:
                        line = "epoch, {}, loss, {}, acc, {}, validate_loss, {}, validate_acc, {}\n".format(
                            begin_epoch + idx, history["loss"][idx], history["acc"][idx],
                            history["val_loss"][idx], history["val_acc"][idx])
                    else:
                        line = "epoch, {}, loss, {}, acc, {}\n".format(
                            begin_epoch + idx, history["loss"][idx], history["acc"][idx])

                    stats_file.write(line)

        except IOError:
            self.log.error("Problem writing to model statistics file %s, check path and permissions", model_file_path)

    def dragon_graphics(self):

        func = lambda x: self.model.predict_probapredict_func(x, verbose=0)[0][0]

        if self.args.checkPoint < 0 or self.args.extendFlag:
            OSMDragonMap(self, self.data.testing(), func).maps(self.args.testDirectory)
            if self.args.extendFlag:
                OSMDragonMap(self, self.data.training(), func).maps(self.args.trainDirectory)

    def similarity_graphics(self):

        def keras_probability(fp, predict_func):
            int_list = []

            for arr in fp:
                int_list.append(arr)

            shape = []
            shape.append(int_list)
            fp_floats = np.array(shape, dtype=float)
            prediction = predict_func(fp_floats, verbose=0)[0][0]  # returns a probability
            return prediction

        func = lambda x: keras_probability(x, self.model.predict_proba)

        if self.args.checkPoint < 0 or self.args.extendFlag:
            OSMSimilarityMap(self, self.data.testing(), func).maps(self.args.testDirectory)
            if self.args.extendFlag:
                OSMSimilarityMap(self, self.data.training(), func).maps(self.args.trainDirectory)


# ===============================================================================
# Keras Pattern Classifier
# ===============================================================================


class KlassSequential(with_metaclass(ModelMetaClass, KerasClassifier)):
    def __init__(self, args, log):
        super(KlassSequential, self).__init__(args, log)

        # Define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".

    # These functions need to be re-defined in all classifier model classes.


    def model_prediction(self, data):
        predictions = self.model.predict_classes(data.input_data(), verbose=0)
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
        score = self.model.evaluate(data.input_data(), binary_labels, verbose=0)
        return score

    def model_probability(self, data):  # probabilities are returned as a numpy.shape = (samples, classes)
        prob =self.model.predict_proba(data.input_data())
        prob_list = list(prob)
        return {"probability": prob_list}

    def train_epoch(self, epoch):

        # shuffle the hold out validation data each epoch.
#        X, y = shuffle(self.data.training().input_data(), self.data.training().target_data())

        classes = self.model_enumerate_classes()
        class_list = self.data.training().target_data()
        index_list = []
        for a_class in class_list:
            index_list.append(classes.index(a_class))
        binary_labels = np_utils.to_categorical(index_list)

        hist = self.model.fit(self.data.training().input_data(), binary_labels, validation_split=self.args.holdOut
                              , nb_epoch=epoch, batch_size=100, verbose=1)

        self.train_history("model_aux.csv", hist.history, epoch)


