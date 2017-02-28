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
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn import cross_validation
from sklearn import tree, metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps


from OSMBase import OSMBaseModel, ModelMetaClass  # The virtual model class.


# A grab-bag of ML techniques implemented in SKLearn. These techniques have not been optimized
# and are offered as an example ease of introducing plug-in classification methods.


class OSMSKLearnSVM(with_metaclass(ModelMetaClass, OSMBaseModel)):

    def __init__(self, args, log):
        super(OSMSKLearnSVM, self).__init__(args, log)     #Edit this and change the class name.
        
# These functions need to be re-defined in all classifier model classes. 

    def model_name(self):
        return "Support Vector Machine (SVM)"  # Model name string.

    def model_postfix(self): # Must be unique for each model.
        return "svm"

    def model_description(self):
        return ("Implements the Support Vector Machine (SVM) Classifier defined in the SKLearn modules.\n"
                " For more information, Google SKLearn and read the documentation.\n"
                " The SVM has not yet been optimized and offered as an example\n"
                " of a plug-in classifier.\n")


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
            graph_file_name = data["ID"][idx] + "_" + self.model_postfix() + ".png"
            graph_path_name = os.path.join(self.args.graphicsDirectory, graph_file_name)
            fig.savefig(graph_path_name, bbox_inches="tight")
            plt.close(fig) # release memory


