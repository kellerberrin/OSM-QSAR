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
from OSMKerasDragon import KlassIonDragon

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

        if self.args.extendFlag:
            OSMSimilarityMap(self, self.data.testing(), func).maps(self.args.testDirectory)
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


class OSMSKLearnNBC(with_metaclass(ModelMetaClass, OSMSKLearnClassifier)):
    def __init__(self, args, log):
        super(OSMSKLearnNBC, self).__init__(args, log)  # Edit this and change the class name.

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


# ================================================================================================
# The Final prediction model used in the OSM competition.
# ================================================================================================

class OSMSKLearnMeta(with_metaclass(ModelMetaClass, OSMSKLearnClassifier)):
    def __init__(self, args, log):
        super(OSMSKLearnMeta, self).__init__(args, log)

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
        return "The OSM SKLearn Competition Classifier"

    def model_postfix(self):  # Must be unique for each model.
        return "osm_sk"

    def model_description(self):
        return (
        "An SKLearn 'off-the-shelf' classification model takes probability functions as input. \n"
        "This classifier model was developed for the OSM molecular ION Activity Competition.\n"
        "It takes the input of an optimal NN that uses the DRAGON data to examine and classify molecular structure\n"
        "and an SKLearn logistic classifier that estimates the molecular potency EC50 potency at 500nMol.\n"
        "The probability maps of these classifiers and then optimally combined to estimate\n"
        "molecular ion activity.")

    def model_define(self):  # Defines the modified sequential class with regularizers defined.

        self.model_meta_train()
        return self.model_arch()

    def model_arch(self):  # Defines the modified sequential class with regularizers defined.
#        model = self.logc_model_define()
        model = self.svmc_model_define()
        return model

    def logc_model_define(self):
        return LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0)

    def dtc_model_define(self):
        return DecisionTreeClassifier(criterion='gini')

    def knnc_model_define(self):
        return KNeighborsClassifier(n_neighbors=5, weights="uniform", algorithm="auto")

    def svmc_model_define(self):
        return OneVsRestClassifier(svm.SVC(kernel=str("rbf"), probability=True, C=1e3, gamma=0.00001))

    def nbc_model_define(self):
                return GaussianNB()
        #        return MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
        #return BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)

    def rfc_model_define(self):
        return RandomForestClassifier(n_estimators=10, criterion='gini')

    def model_prediction(self, data):
#        prediction = self.model.predict(self.input_probability(data))
        classes = self.model_enumerate_classes()
        probs = self.model_probability(data)
        prob_list = probs["probability"]
        class_list = []
        for prob in prob_list:
            if isinstance(prob, list):
                class_list.append(classes[prob.index(max(prob))])
            elif isinstance(prob, np.ndarray):
                klass_index = np.where(prob == prob.max())
                class_list.append(classes[klass_index[0][0]])
            else :
                klass = classes[0] if prob >= 0.5 else classes[1]
                class_list.append(klass)

        return {"prediction": class_list, "actual": data.target_data()}

    def model_probability(self, data):  # probabilities are returned as a numpy.shape = (samples, classes)
        prob = self.model.predict_proba(self.input_probability(data))
        prob_list = list(prob)
        return {"probability": prob_list}

    def model_train(self):
        self.model.fit(self.input_probability(self.data.training()), self.data.training().target_data())

    def input_probability(self, data):

        logc_prob = self.logc.model.predict_proba(data.input_data()["MORGAN2048_5"])

        dnn_dragon_prob = self.dnn_dragon.model.predict_proba(data.input_data()["DRAGON"])
        dnn_dragon_prob = np.asarray(dnn_dragon_prob)

        prob = np.column_stack((dnn_dragon_prob[:, 0], logc_prob[:, 0]))

        return prob

