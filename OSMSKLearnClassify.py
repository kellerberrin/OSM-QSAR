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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree, metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


from OSMBase import ModelMetaClass  # The virtual model class.
from OSMClassify import OSMClassification  # Display and save classifier results.
from OSMModelData import OSMModelData  # specify variable types.
from OSMGraphics import OSMSimilarityMap

# A grab-bag of ML techniques implemented in SKLearn.

######################################################################################################
#
# SKLearn classifier super class - convenient place to put all the common classifier functionality
#
######################################################################################################

class OSMSKLearnClassifier(with_metaclass(ModelMetaClass, OSMClassification)):
    def __init__(self, args, log):
        super(OSMSKLearnClassifier, self).__init__(args, log)  # Edit this and change the class name.

        # define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = { "DEPENDENT" : { "VARIABLE" : "IC50_ACTIVITY", "SHAPE" : [None], "TYPE": OSMModelData.CLASSES }
                         , "INDEPENDENT" : [ { "VARIABLE" : "MORGAN2048_4", "SHAPE": [None], "TYPE": np.float64 } ] }


    def model_train(self):
        # Restrict the SVM to 1 input argument
        self.model.fit(self.data.training().input_data(), self.data.training().target_data())

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




######################################################################################################
#
# Support Vector Machine Implemented as a classifier.
#
######################################################################################################


class OSMSKLearnSVMC(with_metaclass(ModelMetaClass, OSMSKLearnClassifier)):
    def __init__(self, args, log):
        super(OSMSKLearnSVMC, self).__init__(args, log)  # Edit this and change the class name.


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


######################################################################################################
#
# Random Forest Implemented as a classifier.
#
######################################################################################################


class OSMSKLearnRFC(with_metaclass(ModelMetaClass, OSMSKLearnClassifier)):
    def __init__(self, args, log):
        super(OSMSKLearnRFC, self).__init__(args, log)  # Edit this and change the class name.

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "Random Forest (RFC) Classifier"  # Model name string.

    def model_postfix(self):  # Must be unique for each model.
        return "rfc"

    def model_description(self):
        return ("Implements the Random Forest (RFC) Classifier defined in the SKLearn modules.\n"
                " For more information, Google SKLearn and read the documentation.\n")

    def model_define(self):
        return RandomForestClassifier(n_estimators=10, criterion='gini')


######################################################################################################
#
# Naive Bayes. Implemented as a classifier.
#
######################################################################################################


class OSMSKLearnNB(with_metaclass(ModelMetaClass, OSMSKLearnClassifier)):
    def __init__(self, args, log):
        super(OSMSKLearnNB, self).__init__(args, log)  # Edit this and change the class name.

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "Naive Bayes (NBC) Classifier"  # Model name string.

    def model_postfix(self):  # Must be unique for each model.
        return "nbc"

    def model_description(self):
        return ("Implements the Naive Bayes (NBC) Classifier defined in the SKLearn modules.\n"
                " For more information, Google SKLearn and read the documentation.\n")

    def model_define(self):
#        return GaussianNB()
#        return MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
        return BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)



######################################################################################################
#
# Decision Trees. Implemented as a classifier.
#
######################################################################################################


class OSMSKLearnDTC(with_metaclass(ModelMetaClass, OSMSKLearnClassifier)):
    def __init__(self, args, log):
        super(OSMSKLearnDTC, self).__init__(args, log)  # Edit this and change the class name.

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "Decision Tree (DTC) Classifier"  # Model name string.

    def model_postfix(self):  # Must be unique for each model.
        return "dtc"

    def model_description(self):
        return ("Implements the Decision Tree (NBC) Classifier defined in the SKLearn modules.\n"
                " For more information, Google SKLearn and read the documentation.\n")

    def model_define(self):
        return DecisionTreeClassifier(criterion='gini')

######################################################################################################
#
# K Nearest Neighbours. Implemented as a classifier.
#
######################################################################################################


class OSMSKLearnKNNC(with_metaclass(ModelMetaClass, OSMSKLearnClassifier)):
    def __init__(self, args, log):
        super(OSMSKLearnKNNC, self).__init__(args, log)  # Edit this and change the class name.

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "K Nearest Neighbour (KNNC) Classifier"  # Model name string.

    def model_postfix(self):  # Must be unique for each model.
        return "knnc"

    def model_description(self):
        return ("Implements the K Nearest Neighbour (KNNC) Classifier defined in the SKLearn modules.\n"
                " For more information, Google SKLearn and read the documentation.\n")

    def model_define(self):
        return KNeighborsClassifier(n_neighbors=5, weights="uniform", algorithm="auto")

######################################################################################################
#
# K Nearest Neighbours. Implemented as a classifier.
#
######################################################################################################


class OSMSKLearnLOGC(with_metaclass(ModelMetaClass, OSMSKLearnClassifier)):
    def __init__(self, args, log):
        super(OSMSKLearnLOGC, self).__init__(args, log)  # Edit this and change the class name.

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "Logistic (LOGC) Classifier"  # Model name string.

    def model_postfix(self):  # Must be unique for each model.
        return "logc"

    def model_description(self):
        return ("Implements the Logistic (LOGC) Classifier defined in the SKLearn modules.\n"
                " For more information, Google SKLearn and read the documentation.\n")

    def model_define(self):
        return LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0)

