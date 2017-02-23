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
#
import math
import numpy


# ============================================================================
#  A register of classifier models implemented as a dictionary
# ============================================================================


__modelClassRegistry__ = {}    # global dictionary of classification models.


# ================================================================================
# A meta class to automatically register model classes with __modelClassRegistry__
# ================================================================================


class ModelMetaClass(type):

    def __new__(cls, class_name, bases, attrs):

        class_obj = super(ModelMetaClass, cls).__new__(cls, class_name, bases, attrs)
        __modelClassRegistry__[class_name] = class_obj
        return class_obj


# ============================================================================
# Utility functions to enumerate registered classifier model classes
# ============================================================================

def get_model_class(class_name):
    class_obj = None
    if class_name in __modelClassRegistry__:
        class_obj = __modelClassRegistry__[class_name]
    return class_obj

def get_model_instance(class_name, *args):
    return get_model_class(class_name)(*args)

def get_model_method(class_name, method_name):
    method = None
    class_obj = get_model_class(class_name)

    if class_obj is not None:
        if method_name in class_obj.__dict__ :
            method = class_obj.__dict__[method_name]

    return method

# Returns a list of model instances (only models with postfix defined).

def get_model_instances(args, log) :

    model_instances= []

    for class_name in __modelClassRegistry__:
        extfn = get_model_method(class_name, "model_postfix")
        if extfn is not None:
            instance = get_model_instance(class_name, args, log)
            model_instances.append(instance)

    return model_instances

# ============================================================================
# This is a virtual base classification model that is inherited by
# OSM classification models using "from OSMBaseModel import OSMBaseModel".
# See example code in "OSMTemplate.py" and "OSMSequential.py"
# ============================================================================


# class OSMBaseModel(object, metaclass = ModelMetaClass):  # registers all subclasses.
class OSMBaseModel(object):  # registers all subclasses.



    def __init__(self, args, log):

        # Shallow copies of the runtime environment.
        self.log = log
        self.args = args


# ============================================================================        
# These functions need to be defined in derived classes. 
# See "OSMSequential.py" or "OSMTemplate.py".
#        
#    def model_name(self): pass # Name string. Define in derived classes.
#    def model_postfix(self): pass # File postfix string. Define in derived classes.
#    def model_description(self):
#    def model_define(self): pass  # Define in derived classes, returns a model.
#    def model_train(self, model, train): pass # Define in derived classes.
#    def model_read(self, fileName): pass # Define in derived classes, returns a model.
#    def model_write(self, model, fileName): pass # Define in derived classes.
#    def model_prediction(self, model, data): # Define in derived classes.
#        arrayOfPredictions = [-1.0, 0.0, 1.0 ] # Use the model to generate predictions.
#        arrayOfActual = [ 0.0, 1.0, -1.0] # Get an array of actual (predicted) values.
#        return { "prediction" : arrayOfPredictions, "actual" : arrayOfActualValues }
#        
# ============================================================================        

    # Necessary because we need to create the classifier singletons before the args are ready.

    def update_args(self, args):
        self.args = args

    # Perform the classification, generally there should be no need to override this method.

    def classify(self, train, test):

        self.model = self.create_model()

        # Train the model.
        if self.args.retrainFilename != "noretrain" or self.args.loadFilename == "noload":
            self.fit_model(self.model, train)
            self.save_model_file(self.model, self.args.saveFilename)

        # Display training stats
        self.training_stats(self.model, train)

        # Display test results.
        self.display_results(self.model, test)


    def save_model_file(self, model, save_file):
        if save_file != "nosave":
            save_file = save_file + "." + self.model_postfix()
            self.log.info("Saving Trained %s Model in File: %s", self.model_name(), save_file)
            self.model_write(model, save_file)

    def create_model(self): 
        if self.args.loadFilename != "noload" or self.args.retrainFilename != "noretrain":
            
            if self.args.loadFilename != "noload":
                load_file = self.args.loadFilename + "." + self.model_postfix()
            else:
                load_file = self.args.retrainFilename + "." + self.model_postfix()
                
            self.log.info("Loading Pre-Trained %s Model File: %s", self.model_name(), load_file)
            model = self.model_read(load_file)
        else:
            self.log.info("+++++++ Creating %s Model +++++++", self.model_name())
            model = self.model_define()
 
        return model           

    def fit_model(self, model, train):

        self.log.info("Begin Training %s Model", self.model_name())
        self.model_train(model, train)
        self.log.info("End Training %s Model", self.model_name())

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

        predict_array = numpy.array(predict)
        temp_idx = predict_array.argsort()
        predict_ranks = numpy.empty(len(predict_array), int)
        predict_ranks[temp_idx] = numpy.arange(len(predict_array)) + 1

        actual_array = numpy.array(actual)
        temp_idx = actual_array.argsort()
        actual_ranks = numpy.empty(len(actual_array), int)
        actual_ranks[temp_idx] = numpy.arange(len(actual_array)) + 1

# (top 5) Active / Inactive.

        predict_active = ["Active" if x <= 5 else "Inactive" for x in predict_ranks]
        actual_active = ["Active" if x <= 5 else  "Inactive" for x in actual_ranks]
        
# Return the model analysis statistics in a dictionary.
        
        return {"MUE": MUE, "RMSE": RMSE,  "predict_ranks": predict_ranks,
                "actual_ranks": actual_ranks, "predict_active": predict_active,
                "actual_active": actual_active}

# Display the classification results and write to the log file.

    def display_results(self, model, test):
        """Display all the calculated statistics for each model; run"""

        test_predictions = self.model_prediction(model, test)
        test_stats = self.model_accuracy(test_predictions)
                
        self.log.info("Test Compounds pEC50 Mean Unsigned Error (MUE): %f", test_stats["MUE"])
        self.log.info("Test Compounds Results")
        self.log.info("ID, Tested Rank, Pred. Rank, Tested pEC50, Pred. pEC50, Tested Active, Pred. Active")
        self.log.info("===================================================================================")

        for idx in range(len(test["ids"])):
            self.log.info("%s, %d, %d, %f, %f, %s, %s", test["ids"][idx],
                                                        test_stats["actual_ranks"][idx],
                                                        test_stats["predict_ranks"][idx],
                                                        test_predictions["actual"][idx],
                                                        test_predictions["prediction"][idx],
                                                        test_stats["actual_active"][idx],
                                                        test_stats["predict_active"][idx]),




