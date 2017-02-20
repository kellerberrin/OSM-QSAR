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
import math
import numpy

# ============================================================================
# This is a virtual base classification model that is inherited by
# OSM classification models using "from OSMBaseModel import OSMBaseModel".
# See example code in "OSMNewModel.py" and "OSMSequential.py"
# ============================================================================


class OSMBaseModel(object):  # Python 2.7 new style class. Obsolete in Python 3.

    def __init__(self, train, test, args, log):

        # Shallow copies of the runtime environment.
        self.log = log
        self.args = args

        # Compile or load the model.
        self.model = self.create_model()

        # Train the model.
        if self.args.retrainFilename != "noretrain" or self.args.loadFilename == "noload": 
            self.fit_model(self.model, train)
            self.save_model_file(self.model, self.args.saveFilename)

        # Display training stats
        self.training_stats(self.model, train)

        # Display test results.
        self.display_results(self.model, test)

# ============================================================================        
# These functions need to be defined in derived classes. 
# See "OSMSequential.py" or "OSMNewModel.py".         
#        
#    def name(self): pass # Name string. Define in derived classes.
#    def model_file_extension(self): pass # File extension string. Define in derived classes.
#    def define_model(self): pass  # Define in derived classes, returns a model.
#    def train_model(self, model, train): pass # Define in derived classes.
#    def read_model(self, fileName): pass # Define in derived classes, returns a model. 
#    def write_model(self, model, fileName): pass # Define in derived classes.
#    def model_prediction(self, model, data): # Define in derived classes.
#        arrayOfPredictions = [-1.0, 0.0, 1.0 ] # Use the model to generate predictions.
#        arrayOfActual = [ 0.0, 1.0, -1.0] # Get an array of actuals.
#        return { "prediction" : arrayOfPredictions, "actual" : arrayofActualValues }
#        
# ============================================================================        

    def save_model_file(self, model, save_file):
        if save_file != "nosave":
            save_file = save_file + "." + self.model_file_extension()
            self.log.info("Saving Trained %s Model in File: %s", self.name(), save_file)
            self.write_model(model, save_file)

    def create_model(self): 
        if self.args.loadFilename != "noload" or self.args.retrainFilename != "noretrain":
            
            if self.args.loadFilename != "noload":
                load_file = self.args.loadFilename + "." + self.model_file_extension()
            else:
                load_file = self.args.retrainFilename + "." + self.model_file_extension()
                
            self.log.info("Loading Pre-Trained %s Model File: %s", self.name(), load_file)
            model = self.read_model(load_file)
        else:
            self.log.info("+++++++ Creating %s Model +++++++", self.name())
            model = self.define_model()
 
        return model           

    def fit_model(self, model, train):

        self.log.info("Begin Training %s Model", self.name())
        self.train_model(model, train)
        self.log.info("End Training %s Model", self.name())

    def training_stats(self, model, train):
    
        self.trainPredictions = self.model_prediction(model, train)   # Returns a dict. with "prediction" and "actual"
        self.trainStats = self.model_accuracy(self.trainPredictions)  # Returns a dictionary of accuracy tests.
        self.log.info("Training Compounds pEC50 Mean Unsigned Error (MUE): %f", self.trainStats["MUE"])

    def model_accuracy(self, predictions):

        predict = predictions["prediction"]
        actual = predictions["actual"] 

        MUE = 0
        RMSE = 0
        for i in range(len(predict)):
            diff = abs(predict[i]-actual[i])
            MUE += diff
            RMSE = diff * diff            

        MUE = MUE / len(predict)
        RMSE = RMSE / len(predict)
        RMSE = math.sqrt(RMSE) 
        
        
# Sort rankings.

        predarray = numpy.array(predict)
        tempidx = predarray.argsort()
        predranks = numpy.empty(len(predarray), int)
        predranks[tempidx] = numpy.arange(len(predarray)) + 1

        potentarray = numpy.array(actual)
        tempidx = potentarray.argsort()
        potentranks = numpy.empty(len(potentarray), int)
        potentranks[tempidx] = numpy.arange(len(potentarray)) + 1

# (top 5) Active / Inactive.

        predactive = ["Active" if x <= 5 else "Inactive" for x in predranks]
        potentactive = ["Active" if x <= 5 else  "Inactive" for x in potentranks]
        
# Return the model analysis statisitics in a dictionary.        
        
        return {"MUE": MUE, "RMSE": RMSE,  "predranks": predranks,
                "potentranks": potentranks, "predactive": predactive,
                "potentactive": potentactive}

# Display the classification results and write to the log file.

    def display_results(self, model, test):
        """Display all the calculated statistics for each model; run"""

        self.testPredictions = self.model_prediction(model, test)
        self.testStats = self.model_accuracy(self.testPredictions)
                
        self.log.info("Test Compounds pEC50 Mean Unsigned Error (MUE): %f", self.testStats["MUE"])
        self.log.info("Test Compounds Results")
        self.log.info("ID, Tested Rank, Pred. Rank, Tested pEC50, Pred. pEC50, Tested Active, Pred. Active")
        self.log.info("===================================================================================")

        for idx in range(len(test["ids"])):
            self.log.info("%s, %d, %d, %f, %f, %s, %s", test["ids"][idx],
                                                        self.testStats["potentranks"][idx],
                                                        self.testStats["predranks"][idx],
                                                        self.testPredictions["actual"][idx],
                                                        self.testPredictions["prediction"][idx],
                                                        self.testStats["potentactive"][idx],
                                                        self.testStats["predactive"][idx]),




