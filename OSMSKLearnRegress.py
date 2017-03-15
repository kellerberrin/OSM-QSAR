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
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree, metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


from OSMBase import ModelMetaClass  # The virtual model class.
from OSMRegression import OSMRegression  # Display and save regression results.
from OSMModelData import OSMModelData  # specify variable types.
from OSMGraphics import OSMSimilarityMap

# A grab-bag of ML techniques implemented in SKLearn.


######################################################################################################
#
# SKLearn regression super class - convenient place to put all the common regression functionality
#
######################################################################################################

class OSMSKLearnRegression(with_metaclass(ModelMetaClass, OSMRegression)):
    def __init__(self, args, log):
        super(OSMSKLearnRegression, self).__init__(args, log)  # Edit this and change the class name.

        # define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = { "DEPENDENT" : { "VARIABLE" : "pIC50", "SHAPE" : [1], "TYPE": OSMModelData.FLOAT64 }
                 , "INDEPENDENT" : [ { "VARIABLE" : "MORGAN2048_4", "SHAPE": [None], "TYPE": OSMModelData.FLOAT64 } ] }

    def model_train(self):
        self.model.fit(self.data.training().input_data(), self.data.training().target_data())

    def model_prediction(self, data):
        prediction = self.model.predict(data.input_data())
        return {"prediction": prediction, "actual": data.target_data()}

######################################################################################################
#
# Optional member functions.
#
######################################################################################################

    def model_graphics(self):

        def regression_probability(fp, predict_func):
            int_list = []

            for arr in fp:
                int_list.append(arr)

            shape = []
            shape.append(int_list)
            fp_floats = np.array(shape, dtype=float)
            prediction = predict_func(fp_floats)[0]  # returns a prediction (not probability)
            return prediction * -1  # Flip the sign, -ve is good.

        func = lambda x: regression_probability(x, self.model.predict)

        OSMSimilarityMap(self, self.data.testing(), func).maps(self.args.testDirectory)
        if self.args.extendFlag:
            OSMSimilarityMap(self, self.data.training(), func).maps(self.args.trainDirectory)

    ######################################################################################################
    #
    # Support Vector Machine Implemented as a regression.
    #
    ######################################################################################################


class OSMSKLearnSLR(with_metaclass(ModelMetaClass, OSMSKLearnRegression)):
    def __init__(self, args, log):
        super(OSMSKLearnSLR, self).__init__(args, log)  # Edit this and change the class name.

    # These functions need to be re-defined in all regression model classes.

    def model_name(self):
        return "Simple Linear Regression (SLR), Regression"  # Model name string.

    def model_postfix(self):  # Must be unique for each model.
        return "slr"

    def model_description(self):
        return ("Implements a Simple (Least Squares) (SVM) Regression defined in the SKLearn modules.\n"
                " For more information, Google SKLearn and read the documentation.\n")

    def model_define(self):
        return LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)


######################################################################################################
#
# Support Vector Machine Implemented as a regression.
#
######################################################################################################


class OSMSKLearnSVMR(with_metaclass(ModelMetaClass, OSMSKLearnRegression)):
    def __init__(self, args, log):
        super(OSMSKLearnSVMR, self).__init__(args, log)  # Edit this and change the class name.

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


######################################################################################################
#
# Decision Tree Implemented as a regression.
#
######################################################################################################


class OSMSKLearnDTR(with_metaclass(ModelMetaClass, OSMSKLearnRegression)):
    def __init__(self, args, log):
        super(OSMSKLearnDTR, self).__init__(args, log)  # Edit this and change the class name.

    # These functions need to be re-defined in all regression model classes.

    def model_name(self):
        return "Decision Tree (DTR), Regression"  # Model name string.

    def model_postfix(self):  # Must be unique for each model.
        return "dtr"

    def model_description(self):
        return ("Implements the Decision (DTR) Regression defined in the SKLearn modules.\n"
                " For more information, Google SKLearn and read the documentation.\n")

    def model_define(self):
        return DecisionTreeRegressor(criterion="mae")


