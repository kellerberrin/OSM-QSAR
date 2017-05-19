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
from __future__ import absolute_import, division, print_function, unicode_literals

from OSMModelData import OSMModelData

# ============================================================================
#  A register of classifier models implemented as a dictionary
# ============================================================================


__modelClassRegistry__ = {}  # global dictionary of classification models.


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
        if method_name in class_obj.__dict__:
            method = class_obj.__dict__[method_name]

    return method


# Returns a list of model instances (only models with postfix defined).

def get_model_instances(args, log):
    model_instances = []

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


class OSMBaseModel(object):
    def __init__(self, args, log):

        # Shallow copies of the runtime environment.
        self.log = log
        self.args = args

    #####################################################################################
    #
    # Local member functions that call virtual member functions defined
    # elsewhere in the object hierarchy. All functions prefixed "model_" are virtual.
    #
    #####################################################################################

    # Perform the classification, this is the mainline function.

    def classify(self, data):

        self.initialize(data)
        self.model_write()

    def initialize(self, data):

        self.raw_data = data    #The entire dataset for recursive models.
        self.data = OSMModelData(self.args, self.log, self, data) # create a "model-centric" view of the data.
        if self.args.shuffle >= 0:
            self.data.stratified_crossval(self.args.shuffle)  # shuffle the test and training data if flag set.
        self.model = self.create_model()

        self.log.info("Begin Training %s Model", self.model_name())
        self.model_train()
        self.log.info("End Training %s Model", self.model_name())
        self.model_classification_results()
        self.model_training_summary()

    def create_model(self):
        if self.args.loadFilename != "noload":
            model = self.model_read()
        else:
            self.log.info("+++++++ Creating %s Model +++++++", self.model_name())
            model = self.model_define()

        return model

    def model_arguments(self): return self.arguments

    #####################################################################################
    #
    # Virtual member functions redefined elsewhere in the object hierarchy
    #
    #####################################################################################

    def model_is_regression(self): return False   # re-defined in OSMRegression

    def model_is_classifier(self): return False   # re-defined in  OSMClassification

    def model_is_unsupervised(self): return False   # re-defined in  OSMUnsupervised

    # Default for any model without graphics functions.
    def model_graphics(self): pass

    # Redefine these if model I/O is defined.
    def model_write(self): pass

    def model_read(self): return self.model_define()

    def model_epochs(self): return 0

    def model_evaluate(self, data): return []

    def model_training_summary(self): pass

    def model_analytics(self, data): return None
