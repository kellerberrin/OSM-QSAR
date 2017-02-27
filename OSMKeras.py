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

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2, activity_l2
from keras.models import load_model
from keras.constraints import maxnorm
from keras.optimizers import SGD

from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps

from OSMBase import OSMBaseModel, ModelMetaClass  # The virtual model class.


# ===============================================================================
# Base class for the Keras neural network classifiers.
# ===============================================================================

class KerasClassifier(OSMBaseModel):

    def __init__(self, args, log):
        super(KerasClassifier, self).__init__(args, log)

    def model_read(self, file_name):
        return load_model(file_name)

    def model_write(self, model, file_name):
        model.save(file_name)

    def model_train(self, model, train):
    
        if self.args.checkPoint > 0 and self.args.epoch > self.args.checkPoint:
    
            remaining_epochs = self.args.epoch
            while remaining_epochs > 0:

                epochs = self.args.checkPoint if remaining_epochs > self.args.checkPoint else remaining_epochs
                remaining_epochs = remaining_epochs - epochs

                self.keras_train_epoch(model, train, epochs)
                self.save_model_file(model, self.args.saveFilename)

        elif self.args.epoch > 0:
        
            self.keras_train_epoch(model, train, self.args.epoch)
              
        else:
        
            self.keras_train_default(model, train)


    def model_graphics(self, model, train, test):
        self.model_similarity(model, test)      #Only generate similarity maps for the test data.


# ===============================================================================
# The sequential neural net class developed by Vito Spadavecchio.
# ===============================================================================

class SequentialModel(with_metaclass(ModelMetaClass, KerasClassifier)):


    def __init__(self, args, log):
        super(SequentialModel, self).__init__(args, log)

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "Sequential"


    def model_postfix(self):  # Must be unique for each model.
        return "seq"


    def model_description(self):
        return ("A KERAS (TensorFlow) based Neural Network classifier developed by Vito Spadavecchio.\n"
                "The classifier uses 1024 bit Morgan molecular fingerprints in a single layer fully connected NN.")


    def model_define(self): # Defines the sequential class with without regularizer.
    
        model = Sequential()

        model.add(Dense(1024, input_dim=1024, init='uniform', activation='relu'))
        model.add(Dropout(0.2, input_shape=(1024,)))

        model.add(Dense(1, init='normal'))
        model.compile(loss='mean_absolute_error', optimizer='Adam', metrics=['accuracy'])

        return model

    def model_prediction(self, model, data):
        predictions = model.predict(data["MORGAN1024"], verbose=0)
        predictions_array = predictions.flatten()
        return {"prediction": predictions_array, "actual": data["pEC50"] }

    def keras_train_epoch(self, model, train, epoch):
        model.fit(train["MORGAN1024"], train["pEC50"], nb_epoch=epoch, batch_size=45, verbose=1)

    def keras_train_default(self, model, train):
        model.fit(train["MORGAN1024"], train["pEC50"], nb_epoch=1000, batch_size=45, verbose=1)


######################################################################################################
#
# Optional member functions.
#
######################################################################################################


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
            return prob_func(fp_floats, verbose=0)[0][0]

# Ensure that we are using 1024 bit morgan fingerprints.
        def get_fingerprint(mol, atom):
            return SimilarityMaps.GetMorganFingerprint(mol, atom, 4, 'bv', 1024)

        for idx in range(len(data["SMILE"])):
            mol = Chem.MolFromSmiles(data["SMILE"][idx])
            fig, weight = SimilarityMaps.GetSimilarityMapForModel(mol,
                                                                  get_fingerprint,
                                                                  lambda x: get_probability(x, model.predict_proba),
                                                                  colorMap=cm.bwr)
            fig.savefig(os.path.join(self.args.graphicsDirectory,data["ID"][idx] + ".png"), bbox_inches="tight")


# ===============================================================================
# Modified sequential class is a multi layer neural network.
# ===============================================================================

class ModifiedSequential(with_metaclass(ModelMetaClass, KerasClassifier)):


    def __init__(self, args, log):
        super(ModifiedSequential, self).__init__(args, log)

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
        model.add(Dense(2048, input_dim=2048, init='uniform', activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(0.3, input_shape=(2048,)))
        model.add(Dense(30, init='normal', activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(0.3, input_shape=(30,)))
        model.add(Dense(1, init='normal', activation='tanh'))
        model.add(Dense(1, init='normal', activation='linear'))
        sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)

        model.compile(loss='mean_absolute_error', optimizer=sgd, metrics=['accuracy'])
#    	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
#        model.compile(loss='mean_absolute_error', optimizer='Adam', metrics=['accuracy'])

        return model


    def model_prediction(self, model, data):
        predictions = model.predict(data["MORGAN2048"], verbose=0)
        predictions_array = predictions.flatten()
        return {"prediction": predictions_array, "actual": data["pEC50"]}


    def keras_train_epoch(self, model, train, epoch):
        model.fit(train["MORGAN2048"], train["pEC50"], nb_epoch=epoch, batch_size=45, verbose=1)


    def keras_train_default(self, model, train):  # Reduced number of default training epoches.
        model.fit(train["MORGAN2048"], train["pEC50"], nb_epoch=200, batch_size=45, verbose=1)

######################################################################################################
#
# Optional member functions.
#
######################################################################################################

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
            return prob_func(fp_floats, verbose=0)[0][0]

        # Ensure that we are using 2048 bit morgan fingerprints.
        def get_fingerprint(mol, atom):
            return SimilarityMaps.GetMorganFingerprint(mol, atom, 2, 'bv', 2048)

        for idx in range(len(data["SMILE"])):
            mol = Chem.MolFromSmiles(data["SMILE"][idx])
            fig, weight = SimilarityMaps.GetSimilarityMapForModel(mol,
                                                                  get_fingerprint,
                                                                  lambda x: get_probability(x, model.predict_proba),
                                                                  colorMap=cm.bwr)
            fig.savefig(os.path.join(self.args.graphicsDirectory,data["ID"][idx] + ".png"), bbox_inches="tight")

