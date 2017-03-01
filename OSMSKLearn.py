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

import numpy
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

from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps

from OSMBase import ModelMetaClass  # The virtual model class.
from OSMRegression import OSMRegression  # Display and save regression results.
from OSMClassify import OSMClassification  # Display and save classifier results.

# A grab-bag of ML techniques implemented in SKLearn.

######################################################################################################
#
# Support Vector Machine Implemented as a regressor.
#
######################################################################################################

class OSMSKLearnSVMR(with_metaclass(ModelMetaClass, OSMRegression)):

    def __init__(self, args, log):
        super(OSMSKLearnSVMR, self).__init__(args, log)     #Edit this and change the class name.
        
# These functions need to be re-defined in all classifier model classes. 

    def model_name(self):
        return "Support Vector Machine (SVM), Regression"  # Model name string.

    def model_postfix(self): # Must be unique for each model.
        return "svmr"

    def model_description(self):
        return ("Implements the Support Vector Machine (SVM) Classifier defined in the SKLearn modules.\n"
                " This SVM (postfix svmr) is configured as a regression classifier."
                " For more information, Google SKLearn and read the documentation.\n")


    def model_define(self):
        return svm.SVR(kernel='rbf', C=1e3, gamma=0.00001)


    def model_train(self, model, train):
        model.fit(train["MORGAN2048"], train["pEC50"])

    # should return a model, can just return model_define() if there is no model file.
    def model_read(self, file_name):
        self.log.warn("%s model does not save to a model file, a new model was created", self.model_name())
        return self.model_define()

    def model_write(self, model, file_name): pass

    def model_prediction(self, model, data):
        prediction = model.predict(data["MORGAN2048"])
        return {"prediction": prediction, "actual": data["pEC50"] }


######################################################################################################
#
# Optional member functions.
#
######################################################################################################


    def model_graphics(self, model, train, test):
        self.model_similarity(model, test)      #Only generate similarity maps for the test data.


# Generate the png similarity diagrams for the test compounds.
    def model_similarity(self, model, data):

        self.log.info("Generating Similarity Diagrams .......")

        def get_probability(fp, prob_func):
            int_list = []

            for arr in fp:
                int_list.append(arr)

            shape = []
            shape.append(int_list)
            fp_floats = numpy.array(shape, dtype=float)
            prediction = prob_func(fp_floats)[0]   #returns an pEC50 prediction (not probability)
            return prediction * -1.0 # Flip the sign, -ve is good.

        # Ensure that we are using 2048 bit morgan fingerprints.
        def get_fingerprint(mol, atom):
            return SimilarityMaps.GetMorganFingerprint(mol, atom, 2, 'bv', 2048)

        for idx in range(len(data["SMILE"])):
            mol = Chem.MolFromSmiles(data["SMILE"][idx])
            fig, weight = SimilarityMaps.GetSimilarityMapForModel(mol,
                                                                  get_fingerprint,
                                                                  lambda x: get_probability(x, model.predict),
                                                                  colorMap=cm.bwr)
            graph_file_name = data["ID"][idx] + "_sim_" + self.model_postfix() + ".png"
            graph_path_name = os.path.join(self.args.graphicsDirectory, graph_file_name)
            fig.savefig(graph_path_name, bbox_inches="tight")
            plt.close(fig) # release memory


######################################################################################################
#
# Support Vector Machine Implemented as a classifier.
#
######################################################################################################


class OSMSKLearnSVMC(with_metaclass(ModelMetaClass, OSMClassification)):

    def __init__(self, args, log):
        super(OSMSKLearnSVMC, self).__init__(args, log)  # Edit this and change the class name.

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "Support Vector Machine (SVM), Classifier"  # Model name string.

    def model_postfix(self):  # Must be unique for each model.
        return "svmc"

    def model_description(self):
        return ("Implements the Support Vector Machine (SVM) Classifier defined in the SKLearn modules.\n"
                " This SVM (postfix svmc) is configured as a label classifier."
                " For more information, Google SKLearn and read the documentation.\n")

    def model_define(self):
        return OneVsRestClassifier(svm.SVC(kernel='rbf', C=1e3, gamma=0.00001, probability=True))

    def model_train(self, model, train):
        model.fit(train["MORGAN2048"], self.model_classify_pEC50(train["pEC50"])) # convert to labels, then train.

    # should return a model, can just return model_define() if there is no model file.
    def model_read(self, file_name):
        self.log.warn("%s model does not save to a model file, a new model was created", self.model_name())
        return self.model_define()

    def model_write(self, model, file_name):
        pass

    def model_prediction(self, model, data):
        prediction = model.predict(data["MORGAN2048"])
        return {"prediction": prediction, "actual": self.model_classify_pEC50(data["pEC50"])} #Convert to labels.


    ######################################################################################################
    #
    # Optional member functions.
    #
    ######################################################################################################


    def model_graphics(self, model, train, test):
        self.model_similarity(model, test)  # Only generate similarity maps for the test data.
#        self.model_ROC(model, test)

    # Generate the png similarity diagrams for the test compounds.
    def model_similarity(self, model, data):

        self.log.info("Generating Similarity Diagrams .......")

        def get_probability(fp, prob_func):
            int_list = []

            for arr in fp:
                int_list.append(arr)

            shape = []
            shape.append(int_list)
            fp_floats = numpy.array(shape, dtype=float)
            active_prob = prob_func(fp_floats)[0]  # returns an "active" probability (first element of the prob array).
            return active_prob

        # Ensure that we are using 2048 bit morgan fingerprints.
        def get_fingerprint(mol, atom):
            return SimilarityMaps.GetMorganFingerprint(mol, atom, 2, 'bv', 2048)

        for idx in range(len(data["SMILE"])):
            mol = Chem.MolFromSmiles(data["SMILE"][idx])
            fig, weight = SimilarityMaps.GetSimilarityMapForModel(mol,
                                                                  get_fingerprint,
                                                                  lambda x: get_probability(x, model.predict_proba),
                                                                  colorMap=cm.bwr)
            graph_file_name = data["ID"][idx] + "_sim_" + self.model_postfix() + ".png"
            graph_path_name = os.path.join(self.args.graphicsDirectory, graph_file_name)
            fig.savefig(graph_path_name, bbox_inches="tight")
            plt.close(fig)  # release memory


    # Plot of a ROC curve for a specific class

    def model_ROC(self, model, data):

        plt.figure()
        lw = 2
        plt.plot(fpr[2], tpr[2], color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic "svmc"')
        plt.legend(loc="lower right")
        plt.show()
        graph_file_name = "svmc_roc_" + self.model_postfix() + ".png"
        graph_path_name = os.path.join(self.args.graphicsDirectory, graph_file_name)
        plt.savefig(graph_path_name, bbox_inches="tight")
        plt.close()  # release memory

