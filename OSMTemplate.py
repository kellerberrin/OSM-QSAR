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

import numpy

from OSMBase import ModelMetaClass  # The virtual model class.
from OSMRegression import OSMRegression  # Display and save regression results.
from OSMClassify import OSMClassification  # Display and save classification results.
from OSMUtility import OSMUtility


class OSMRegressionTemplate(with_metaclass(ModelMetaClass, OSMRegression)):  # Edit this and change the class name
    # This classification is by regression therefore inherit "with_metaclass(ModelMetaClass, OSMRegression)".

    def __init__(self, args, log):
        super(OSMRegressionTemplate, self).__init__(args, log)  # Edit this and change the class name.

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "REGRESSION Template Model (unimplemented)"  # Model name string.

    def model_postfix(self):  # Must be unique for each model.
        return "tmpr"

    def model_description(self):
        return ("A template object of an OSM molecular REGRESSION model.\n"
                "To implement a new OSM regression model take the following steps:\n"
                ' 1. Copy "OSMTemplate.py" to "OSMMyNewModel.py"\n'
                ' 2. Change the class name to "OSMMyNewModel".\n'
                ' 3. Redefine the member functions that begin with "model_".\n'
                ' 4. Add the line "from OSMMyNewModel import OSMMYNewModel" to "OSM_QSAR.py".\n'
                " That's all. All the statistics, analytics and (implemented) graphics functionality\n"
                " are now used by your model.")

    def model_define(self):
        return None  # Should return a model.

    def model_train(self, model, train): pass

    def model_read(self, file_name):
        return None  # should return a model, can just return model_define() if there is no model file.

    def model_write(self, model, file_name): pass

    def model_prediction(self, model, data):
        return {"prediction": data["pEC50"], "actual": data["pEC50"]}

######################################################################################################
#
# Optional member functions.
#
######################################################################################################


class OSMClassificationTemplate(with_metaclass(ModelMetaClass, OSMClassification)):  # Edit this and change the class name
    # This is a classifier so inherit "with_metaclass(ModelMetaClass, OSMClassification)".

    def __init__(self, args, log):
        super(OSMClassificationTemplate, self).__init__(args, log)  # Edit this and change the class name.

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "CLASSIFICATION Template Model (unimplemented)"  # Model name string.

    def model_postfix(self):  # Must be unique for each model.
        return "tmpc"

    def model_description(self):
        return ("A template object of an OSM molecular CLASSIFICATION model.\n"
                "To implement a new OSM classification model take the following steps:\n"
                ' 1. Copy "OSMTemplate.py" to "OSMMyNewModel.py"\n'
                ' 2. Change the class name to "OSMMyNewModel".\n'
                ' 3. Redefine the member functions that begin with "model_".\n'
                ' 4. Add the line "from OSMMyNewModel import OSMMYNewModel" to "OSM_QSAR.py".\n'
                " That's all. All the statistics, analytics and (implemented) graphics functionality\n"
                " are now used by your model.")

    def model_define(self):
        return None  # Should return a model.

    def model_train(self, model, train): pass

    def model_read(self, file_name):
        return None  # should return a model, can just return model_define() if there is no model file.

    def model_write(self, model, file_name): pass

    def model_prediction(self, model, data): # prediction and actual are returned as one hot vectors.
        return {"prediction": OSMUtility.data_classify(data["pEC50"], self.args.activeNmols)
            , "actual": OSMUtility.data_classify(data["pEC50"], self.args.activeNmols)}

    def model_probability(self, model, data):  # probabilities are returned as a numpy.shape = (samples, classes)
        probability = numpy.zeros((len(data["pEC50"]), len(OSMUtility.enumerate_classes(self.args.activeNmols))))
        probability[:, 0] = 1.0
        return {"probability": probability}


######################################################################################################
#
# Optional member functions.
#
######################################################################################################


