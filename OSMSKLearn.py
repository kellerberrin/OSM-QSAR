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

import os
import sys

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn import tree, metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


from OSMBase import ModelMetaClass  # The virtual model class.
from OSMRegression import OSMRegression  # Display and save regression results.
from OSMClassify import OSMClassification  # Display and save classifier results.

from OSMGraphics import OSMSimilarityMap

# A grab-bag of ML techniques implemented in SKLearn.

######################################################################################################
#
# Support Vector Machine Implemented as a regression.
#
######################################################################################################

class OSMSKLearnSVMR(with_metaclass(ModelMetaClass, OSMRegression)):
    def __init__(self, args, log):
        super(OSMSKLearnSVMR, self).__init__(args, log)  # Edit this and change the class name.

        # define the model data view.
        self.arguments = { "DEPENDENT" : { "VARIABLE" : "pIC50", "SHAPE" : [1], "TYPE": np.float64 }
                         , "INDEPENDENT" : [ { "VARIABLE" : "MORGAN2048", "SHAPE": [None], "TYPE": np.float64 } ] }

    # These functions need to be re-defined in all regression model classes.

    def model_name(self):
        return "Support Vector Machine (SVM), Regression"  # Model name string.

    def model_postfix(self):  # Must be unique for each model.
        return "svmr"

    def model_description(self):
        return ("Implements the Support Vector Machine (SVM) Regression defined in the SKLearn modules.\n"
                " This SVM (postfix svmr) is configured as a regression classifier.\n"
                " For more information, Google SKLearn and read the documentation.\n")

    def model_define(self):
        return svm.SVR(kernel=str("rbf"), C=1e3, gamma=0.00001)

    def model_train(self):
        self.model.fit(self.data.training().input_data(), self.data.training().target_data())

    # Just returns a model_define() and complains since there is no model file operation defined.
    def model_read(self, file_name):
        self.log.warn("%s model does not save to a model file, a new model was created", self.model_name())
        return self.model_define()

    def model_write(self, file_name):
        self.log.warn("%s model write function not defined.", self.model_name())
        return

    def model_prediction(self, data):
        prediction = self.model.predict(data.input_data())
        return {"prediction": prediction, "actual": data.target_data()}

######################################################################################################
#
# Optional member functions.
#
######################################################################################################

    def model_graphics(self):

        def svmr_probability(fp, predict_func):
            int_list = []

            for arr in fp:
                int_list.append(arr)

            shape = []
            shape.append(int_list)
            fp_floats = np.array(shape, dtype=float)
            prediction = predict_func(fp_floats)[0]  # returns a prediction (not probability)
            return prediction * -1  # Flip the sign, -ve is good.

        func = lambda x: svmr_probability(x, self.model.predict)

        OSMSimilarityMap(self, self.data.testing(), func).maps(self.args.testDirectory)
        if self.args.extendFlag:
            OSMSimilarityMap(self, self.data.training(), func).maps(self.args.trainDirectory)



######################################################################################################
#
# Support Vector Machine Implemented as a classifier.
#
######################################################################################################


class OSMSKLearnSVMC(with_metaclass(ModelMetaClass, OSMClassification)):
    def __init__(self, args, log):
        super(OSMSKLearnSVMC, self).__init__(args, log)  # Edit this and change the class name.

        # define the model data view.
        self.arguments = { "DEPENDENT" : { "VARIABLE" : "ION_ACTIVITY", "SHAPE" : [1], "TYPE": np.str }
                         , "INDEPENDENT" : [ { "VARIABLE" : "MORGAN2048", "SHAPE": [None], "TYPE": np.float64 } ] }

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "Support Vector Machine (SVM) Classifier"  # Model name string.

    def model_postfix(self):  # Must be unique for each model.
        return "svmc"

    def model_description(self):
        return ("Implements the Support Vector Machine (SVM) Classifier defined in the SKLearn modules.\n"
                " This SVM (postfix svmc) is configured as a label classifier.\n"
                " For more information, Google SKLearn and read the documentation.\n")

    def model_define(self):
        return OneVsRestClassifier(svm.SVC(kernel=str("rbf"), probability=True, C=1e3, gamma=0.00001))

    def model_train(self):
        # Restrict the SVM to 1 input argument
        self.model.fit(self.data.training().input_data(), self.data.training().target_data())

    # Just returns a model_define() and complains since there is no model file operation defined.
    def model_read(self, file_name):
        self.log.warn("%s model does not save to a model file, a new model was created", self.model_name())
        return self.model_define()

    def model_write(self, file_name):
        self.log.warn("%s model write function not defined.", self.model_name())
        return

    def model_prediction(self, data):
        prediction = self.model.predict(data.input_data())
        return {"prediction": prediction, "actual": data.target_data()}

    def model_probability(self, data):  # probabilities are returned as a numpy.shape = (samples, classes)
        probability = self.model.predict_proba(data.input_data())
        return {"probability": probability}

    ######################################################################################################
    #
    # Optional member functions.
    #
    ######################################################################################################

    def model_graphics(self):

        def classifier_probability(fp, prob_func):
            int_list = []

            for arr in fp:
                int_list.append(arr)

            shape = [int_list]
            fp_floats = np.array(shape, dtype=float)
            active_prob = prob_func(fp_floats)[0][0]  # returns an "active" probability (element[0]).
            return active_prob

        func = lambda x: classifier_probability(x, self.model.predict_proba)

        OSMSimilarityMap(self, self.data.testing(), func).maps(self.args.testDirectory)
        if self.args.extendFlag:
            OSMSimilarityMap(self, self.data.training(), func).maps(self.args.trainDirectory)

