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

import sys

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2, activity_l2
from keras.models import load_model
from keras.constraints import maxnorm
from keras.optimizers import SGD


from OSMBase import ModelMetaClass  # The virtual model class.
from OSMRegression import OSMRegression  # Display and save regression results.
from OSMGraphics import OSMSimilarityMap
from OSMModelData import OSMModelData
from OSMIterative import OSMIterative


# ===============================================================================
# Base class for the Keras neural network classifiers.
# ===============================================================================

class KerasClassifier(OSMRegression):

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

        def keras_probability(fp, predict_func):
            int_list = []

            for arr in fp:
                int_list.append(arr)

            shape = []
            shape.append(int_list)
            fp_floats = np.array(shape, dtype=float)
            prediction = predict_func(fp_floats, verbose=0)[0][0]  # returns a prediction (not probability)
            return prediction * -1  # Flip the sign, -ve is good.

        func = lambda x: keras_probability(x, self.model.predict)

        if self.args.checkPoint < 0 or self.args.extendFlag:
            OSMSimilarityMap(self, self.data.testing(), func).maps(self.args.testDirectory)
            if self.args.extendFlag:
                OSMSimilarityMap(self, self.data.training(), func).maps(self.args.trainDirectory)

# ===============================================================================
# The sequential neural net class developed by Vito Spadavecchio.
# ===============================================================================

class SequentialModel(with_metaclass(ModelMetaClass, KerasClassifier)):

    def __init__(self, args, log):
        super(SequentialModel, self).__init__(args, log)

        # Define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".

        self.arguments = { "DEPENDENT" : { "VARIABLE" : "pIC50", "SHAPE" : [1], "TYPE": OSMModelData.FLOAT64 }
              , "INDEPENDENT" : [ { "VARIABLE" : "MORGAN1024", "SHAPE": [1024], "TYPE": OSMModelData.FLOAT64 } ] }

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "Sequential"

    def model_postfix(self):  # Must be unique for each model.
        return "seq"

    def model_description(self):
        return ("A KERAS (TensorFlow) based Neural Network classifier developed by Vito Spadavecchio.\n"
                "The classifier uses 1024 bit Morgan molecular fingerprints in a single layer fully connected NN.")

    def model_define(self):

        model = Sequential()

        model.add(Dense(1024, input_dim=1024, init="uniform", activation="relu"))
        model.add(Dropout(0.2, input_shape=(1024,)))
        model.add(Dense(1, init="normal"))
        model.compile(loss="mean_absolute_error", optimizer="Adam", metrics=["accuracy"])

        return model

    def model_prediction(self, data):
        predictions = self.model.predict(data.input_data(), verbose=0)
        predictions_array = predictions.flatten()
        return {"prediction": predictions_array, "actual": data.target_data() }

    def train_epoch(self, epoch):
        self.model.fit(self.data.training().input_data(), self.data.training().target_data()
                       , nb_epoch=epoch, batch_size=45, verbose=1)

# ===============================================================================
# Modified sequential class is a multi layer neural network.
# ===============================================================================

class ModifiedSequential(with_metaclass(ModelMetaClass, KerasClassifier)):


    def __init__(self, args, log):
        super(ModifiedSequential, self).__init__(args, log)

        self.default_epochs = 200

        # Define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = { "DEPENDENT" : { "VARIABLE" : "pIC50", "SHAPE" : [1], "TYPE": OSMModelData.FLOAT64 }
              , "INDEPENDENT" : [ { "VARIABLE" : "MORGAN2048", "SHAPE": [2048], "TYPE": OSMModelData.FLOAT64 } ] }

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "Modified Sequential"

    def model_postfix(self):   # Must be unique for each model.
        return "mod"

    def model_description(self):
        return ("A KERAS (TensorFlow) multi-layer Neural Network classification model. \n"
                "This classifier analyzes 2048 bit Morgan molecular fingerprints.")

    def model_define(self): # Defines the modified sequential class with regularizers defined.
    
        model = Sequential()

#        model.add(Dense(2048, input_dim=2048, init='uniform', activation='relu',W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
        model.add(Dense(2048, input_dim=2048, init="uniform", activation="relu", W_constraint=maxnorm(3)))
        model.add(Dropout(0.3, input_shape=(2048,)))
        model.add(Dense(30, init="normal", activation="relu", W_constraint=maxnorm(3)))
        model.add(Dropout(0.3, input_shape=(30,)))
        model.add(Dense(1, init="normal", activation="tanh"))
        model.add(Dense(1, init="normal", activation="linear"))
        sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)

        model.compile(loss='mean_absolute_error', optimizer="Adam", metrics=['accuracy'])
#    	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
#        model.compile(loss='mean_absolute_error', optimizer='Adam', metrics=['accuracy'])

        return model

    def model_prediction(self,data):
        predictions = self.model.predict(data.input_data(), verbose=0)
        predictions_array = predictions.flatten()
        return {"prediction": predictions_array, "actual": data.target_data()}


    def train_epoch(self, epoch):
        self.model.fit(self.data.training().input_data(), self.data.training().target_data()
                       , nb_epoch=epoch, batch_size=100, verbose=1)

