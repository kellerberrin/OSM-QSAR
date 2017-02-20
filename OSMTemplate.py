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


from OSMBase import OSMBaseModel  # The virtual model class.


# This is a stub object that can be used as a template to create new OSM classifiers.
# 1. Copy "OSMNewModel.py" to 'OSMMyModel.py"
# 2. Redefine the member functions below.
# 3. In the header of "OSMClassify.py" add "from OSMMyModel import OSMMyModelObjectName" 
# 4. Add your new model to the "Classification" object in "OSMClassify.py"  


class OSMNewModel(OSMBaseModel):

    def __init__(self, train, test, args, log):
        super(OSMNewModel, self).__init__(train, test, args, log)
        
# These functions need to be re-defined in all classifier model classes. 
# See "OSMBaseModel.py" and "OSMSequential.py"          
        
    def name(self):
        return "New Model (unimplemented)"  # Model name string.

    def model_file_extension(self):
        return "new"  # File extension string.

    def define_model(self):
        return None  # Should return a model.

    def train_model(self, model, train): pass

    def read_model(self, file_name):
        f = open(file_name, "r")
        self.log.info("Read OSMNewModel file: %s, content: %s", file_name, f.readline())
        return None  # should return a model

    def write_model(self, model, file_name):
        f = open(file_name, "w")
        f.write("OSMNewModel not implemented yet\n")

    def model_prediction(self, model, data):
        return {"prediction": data["pEC50"], "actual": data["pEC50"] }
        

