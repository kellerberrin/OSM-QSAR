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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#
#


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2, activity_l2
from keras.models import load_model
from keras.constraints import maxnorm
from keras.optimizers import SGD

from OSMBase import OSMBaseModel  # The virtual model class.


# ===============================================================================
# Base class for the Keras neural network classifiers.
# ===============================================================================

class KerasClassifier(OSMBaseModel):

    def __init__(self, train, test, args, log):
        super(KerasClassifier, self).__init__(train, test, args, log)


    def read_model(self, fileName):
        return load_model(fileName)
 
    def write_model(self, model, fileName):
        model.save(fileName)

                    
    def model_prediction(self, model, data):
        predictions = model.predict(data["morgan"], verbose=0)
        predictionsArray = predictions.flatten()
        return { "prediction" : predictionsArray, "actual" : data["pEC50"] } 

    def train_model(self, model, train):
    
        if self.args.epoch > 0 and self.args.checkPoint > 0 and self.args.epoch > self.args.checkPoint:      
    
            remainingEpochs = self.args.epoch
            while remainingEpochs > 0:

                epochs = self.args.checkPoint if remainingEpochs > self.args.checkPoint else remainingEpochs
                remainingEpochs = remainingEpochs - epochs

                self.train_epoch(model, train, epochs)
                self.save_model_file(model, self.args.saveFilename)

        elif self.args.epoch > 0:
        
            self.train_epoch(model, train, self.args.epoch)
              
        else:
        
            self.train_default(model, train)

        
    def train_epoch(self, model, train, epoch):        
        model.fit(train["morgan"], train["pEC50"], nb_epoch=epoch, batch_size=45, verbose=1)
            
    

# ===============================================================================
# The sequential neural net class developed by Vito Spadavecchio.
# ===============================================================================

class SequentialModel(KerasClassifier):

    def __init__(self, train, test, args, log):
        super(SequentialModel, self).__init__(train, test, args, log)

    def name(self): 
        return "Sequential"

    def model_file_extension(self):
        return "seq"

    def define_model(self): # Defines the sequential class with without regularizer.
    
        model = Sequential()

        model.add(Dense(1024, input_dim=1024, init='uniform', activation='relu'))
        model.add(Dropout(0.2, input_shape=(1024,)))

        model.add(Dense(1, init='normal'))
        model.compile(loss='mean_absolute_error', optimizer='Adam', metrics=['accuracy'])

        return model

    def train_default(self, model, train):        
        model.fit(train["morgan"], train["pEC50"], nb_epoch=1000, batch_size=45, verbose=1)


# ===============================================================================
# Modified sequential class uses regularizers 
# ===============================================================================

class ModifiedSequential(KerasClassifier):

    def __init__(self, train, test, args, log):
        super(ModifiedSequential, self).__init__(train, test, args, log)

    def name(self): 
        return "Modified Sequential"

    def model_file_extension(self):
        return "mod"

    def define_model(self): # Defines the modified sequential class with regularizers defined.
    
        model = Sequential()

#        model.add(Dense(1024, input_dim=1024, init='uniform', activation='relu',W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)))
        model.add(Dense(1024, input_dim=1024, init='uniform', activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(0.3, input_shape=(1024,)))
        model.add(Dense(30, init='normal', activation='relu', W_constraint=maxnorm(3)))
        model.add(Dropout(0.3, input_shape=(1024,)))
        model.add(Dense(1, init='normal', activation='tanh'))
        model.add(Dense(1, init='normal', activation='linear'))
        sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)

        model.compile(loss='mean_absolute_error', optimizer=sgd, metrics=['accuracy'])
#    	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
#        model.compile(loss='mean_absolute_error', optimizer='Adam', metrics=['accuracy'])

        return model

    def train_default(self, model, train):  # Reduced number of default training epoches.    
        model.fit(train["morgan"], train["pEC50"], nb_epoch=200, batch_size=45, verbose=1)




