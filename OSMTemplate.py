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


from OSMBase import OSMBaseModel, ModelMetaClass  # The virtual model class.


# This is a stub object that can be used as a template to create new OSM classifiers.
# 1. Copy "OSMTemplate.py" to 'OSMMyModel.py"
# 2. Redefine the member functions below.


class OSMTemplateModel(with_metaclass(ModelMetaClass, OSMBaseModel)):   # Edit this and change the class name
# If you inherit from a class that is a subclass of OSMBaseModel then change the base class name.
# For example, if the base class is "OSMMYBase" then change to "with_metaclass(ModelMetaClass, OSMMyBase)".

    def __init__(self, args, log):
        super(OSMTemplateModel, self).__init__(args, log)     #Edit this and change the class name.
        
# These functions need to be re-defined in all classifier model classes. 

    def model_name(self):
        return "Template Model (unimplemented)"  # Model name string.

    def model_postfix(self): # Must be unique for each model.
        return "tmp"

    def model_description(self):
        return ("A template object of an OSM molecular classifier.\n"
                "To implement a new OSM classification model take the following steps:\n"
                ' 1. Copy "OSMTemplate.py" to "OSMMyNewModel.py"\n'
                ' 2. Change the class name to "OSMMyNewModel".\n'
                ' 3. Redefine the member functions that begin with "model_".\n'
                ' 4. add the line "from OSMMyNewModel import OSMMYNewModel" to "OSM_QSAR.py".\n'
                " That's all. All the statistics, analytics and graphics functionality are now used by your model.")

    def model_define(self):
        return None  # Should return a model.

    def model_train(self, model, train): pass

    def model_read(self, file_name):
        return None  # should return a model, can just return model_define() if there is no model file.

    def model_write(self, model, file_name): pass

    def model_prediction(self, model, data):
        return {"prediction": data["pEC50"], "actual": data["pEC50"] }
        

######################################################################################################
#
# Optional member functions.
#
######################################################################################################

# Define a similarity map for the model.
#    def model_similarity(self, model, data): pass
