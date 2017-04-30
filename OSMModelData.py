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

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

import sys
import numpy as np
import pandas as pd
import copy


from OSMProperties import OSMGenerateData


# Because QSAR fields can be assigned to a model at runtime, each model needs to published the variables types it
# will accept.

#self.arguments = { "DEPENDENT" : { "VARIABLE" : "IC50_ACTIVITY", "SHAPE" : [3], "TYPE": OSMDataModel.CLASSES }
#                         , "INDEPENDENT" : [ { "VARIABLE" : "MORGAN2048", "SHAPE": [2048], "TYPE": np.float64 } ] }

# The specification consists of a single variable for the dependent (target) variable and a list of
# independent variables. For models of the OSMUnsupervised class, the dependent variable is None (no dependent
# variable).
#
# Each variable specification consists of 3 fields. "VARIABLE" is the name of the default variable. The variable can
# be modified at runtime using the "--depend" or "--indep" flags (a list of variables with the indep flag).
# "TYPE" is the type of the variable(s) these are "OSMModelData.FLOAT64" ( equivalent to np.float64, floating point),
# "OSMModelData.STRING" (equivalent to np.str, string) and "OSMModelData.CLASSES" (number a patterns - classes - for
# a model that inherits from OSMClassification). Finally, as the name suggests, the "SHAPE" variable determines the
# dimensionality of the variable. "SHAPE" is specified as [Dim1, Dim2, ..., Dimn] for variables of type
# "OSMModelData.FLOAT64" and "OSMModelData.STRING".  For variables specified as "OSMModelData.classes", the "SHAPE"
# variable specifies the number of classes, [classes1, classes2,..., classes] presented to a model that inherits
# from OSMClassification.
#
# "SHAPE" can specified as "SHAPE" : None. In which no shape checking is performed before the variable is
# presented to the model. "SHAPE" can also be specified as "SHAPE : [None, ..., None] which specifies the number
# of dimensions but not the size of each dimension. For example the Support Vector Model Classifier "svmc" has the
# variables specification.
#
#     self.arguments = { "DEPENDENT" : { "VARIABLE" : "ION_ACTIVITY", "SHAPE" : [None], "TYPE": OSMModelData.CLASSES }
#                         , "INDEPENDENT" : [ { "VARIABLE" : "MORGAN2048", "SHAPE": [None], "TYPE": np.float64 } ] }
#
# This means that it an OSMClassification model with a dependent variable that has 1 dimension and accepts any
# number of classes. It also has an independent variable that is a 1 dimensional float (vector) that accepts any
# number of elements.
#
# ===================================================================================================
#
# Presents the 'model-centric' view of the data.
#
# ===================================================================================================

class OSMModelData(object):

    # Specifiy the variable types.
    FLOAT64 = np.float64
    STRING = np.str
    CLASSES = "__CLASSES__"

    def __init__(self, args, log, model, data):

        self.log = log
        self.args = args
        self.model_args = model.model_arguments()
        self.model_df = copy.deepcopy(data.get_data())    # The data object is used elsewhere, leave it unmolested.
        self.dragon_headers = data.get_dragon_headers()
        self.train, self.test = self.setup_model_data()

    def training(self):  # Access training data (returns the adapter class).
        return AccessData(self, self.train)

    def testing(self):   # Access the test (target) data (returns the adapter class).
        return AccessData(self, self.test)

    def all(self):   # Access all the data (returns the adapter class).
        return AccessData(self, self.model_df)

    @staticmethod
    def enumerate_classes(class_column):

        class_set = set(class_column)
        class_list = list(class_set)  # convert back to set
        sorted_class_list = sorted(class_list)  # ascending sort.
        return sorted_class_list

########################################################################################################
# Local member functions.
########################################################################################################


    def setup_model_data(self):
        self.check_model_arguments()
        return self.create_train_test()

    def check_model_arguments(self):
        if self.args.dependVar != "default":
            self.model_args["DEPENDENT"]["VARIABLE"] = self.args.dependVar

        self.check_arg_shape_type(self.model_args["DEPENDENT"])  # raises an exception if any errors.

        if self.args.indepList != "default":

            if self.model_args["INDEPENDENT"][0]["SHAPE"] is None:

                if not self.model_args["INDEPENDENT"][0]["SHAPE"][0] is None:

                    if len(self.args.indepList) != len(self.model_args["INDEPENDENT"] ):
                        self.log.error("Model requires %d independent variables;  %s specified",
                                       len(self.model_args["INDEPENDENT"]), ",".join(self.args.indepList))
                    sys.exit()

            for idx in range(len(self.model_args["INDEPENDENT"])):
                self.model_args["INDEPENDENT"][idx]["VARIABLE"] = self.args.indepList[idx]

        for arg in self.model_args["INDEPENDENT"]:
            self.check_arg_shape_type(arg)  # raises an exception if any errors.

    def check_arg_shape_type(self, arg):

        if arg["VARIABLE"] not in self.model_df.columns:
            self.log.error('Model variable %s not found in data frame, check with "--vars"', arg["VARIABLE"])
            sys.exit()

        self.convert_type(arg)

        if arg["TYPE"] == OSMModelData.CLASSES:
            self.check_classes_shape(arg)
        else:
            self.check_shape(arg)

    def check_shape(self, arg):

        arg_shape =arg["SHAPE"]

        if arg_shape is None: return    # No shape checking

        if len(arg_shape) == 1 and arg_shape[0] == 1:  # Check if we require a scalar

            if not isinstance(self.model_df[arg["VARIABLE"]][0], np.ndarray):  # Numpy must have shape (1,)
                return
            else:
                # Assume we have a list of numpy.arrays, only check the first element
                df_shape = list(self.model_df[arg["VARIABLE"]][0].shape)
                if len(df_shape) == 1 and df_shape[0] == 1:
                    return
                else:
                    self.log.error("Expecting scalar variable %s, got numpy array.", arg["VARIABLE"])
                    sys.exit()

        # Assume we have a list of numpy.arrays, only check the first element
        df_shape = list(self.model_df[arg["VARIABLE"]][0].shape)

        # Check that the shapes have the same dimension

        if len(df_shape) != len(arg_shape):
            self.log.error("Variable %s has dimension %d, expecting dimension %d."
                           , arg["VARIABLE"], len(df_shape), len(arg_shape))
            sys.exit()

        for idx in range(len(arg_shape)):

            if arg_shape[idx] is None: continue

            if arg_shape[idx] != df_shape[idx]:
                self.log.error("Variable %s[%d] has shape %d, model requires %d.", arg["VARIABLE"],
                               idx, df_shape[idx], arg_shape[idx])
                sys.exit()


    def check_classes_shape(self, arg):

        arg_shape =arg["SHAPE"]

        if arg_shape is None: return    # No shape checking

        if len(arg_shape) == 0: return # no shape checking

        if len(arg_shape) == 1:  # Check if a scalar

            if arg_shape[0] == None: return # no shape checking.

            if not isinstance(self.model_df[arg["VARIABLE"]][0], np.ndarray):

                if arg_shape[0] != len(OSMModelData.enumerate_classes(self.model_df[arg["VARIABLE"]])):
                    self.log.error("Class variable %s, has classes %s, expected %d classes"
                                   , arg["VARIABLE"]
                                   , OSMModelData.enumerate_classes(self.model_df[arg["VARIABLE"]])
                                   , arg_shape[0])
                    sys.exit()
                else: return #OK

            else:
                self.log.error("Variable %s is a Numpy array. Arrays of classes not yet implemented.", arg["VARIABLE"])
                sys.exit()
        else:
            self.log.error("Variable %s specifies multidimensional classes, not yet implemented.", arg["VARIABLE"])
            sys.exit()

    def convert_type(self, arg):

        if not isinstance(self.model_df[arg["VARIABLE"]][0], np.ndarray):  # only if scalar otherwise a numpy array.

            if arg["TYPE"] == np.float64: # convert to numeric.
                try:
                    pd.to_numeric(self.model_df[arg["VARIABLE"]])
                    self.model_df.dropna(subset=[arg["VARIABLE"]], inplace=True)  # Delete all NaN rows.
                    self.scale_numeric(arg)

                except ValueError:
                    self.log.error("Problem converting variable %s to floats", arg["VARIABLE"])
                    sys.exit()
            elif arg["TYPE"] == np.str:

                if isinstance(self.model_df[arg["VARIABLE"]][0], np.float64):
                    self.log.error("Variable %s is a float, model specifies a string (class)", arg["VARIABLE"])
                    sys.exit()

                self.model_df[arg["VARIABLE"]].replace("", np.nan, inplace=True) # Convert empty fields to NaNs
            elif arg["TYPE"] == OSMModelData.CLASSES:

                if isinstance(self.model_df[arg["VARIABLE"]][0], np.float64):
                    self.log.error("Variable %s is a float, model specifies a class", arg["VARIABLE"])
                    sys.exit()

                self.model_df[arg["VARIABLE"]].replace("", np.nan, inplace=True) # Convert empty fields to NaNs
            else:
                self.model_df[arg["VARIABLE"]].replace("", np.nan, inplace=True) # Convert empty fields to NaNs

            self.model_df.dropna(subset=[arg["VARIABLE"]], inplace=True) # Delete all NaN rows.

        if self.model_df[arg["VARIABLE"]].shape[0] == 0:
            self.log.error("No valid values in for variable %s (check string or numeric)", arg["VARIABLE"])
            sys.exit()

    def scale_numeric(self, arg):
        narray = self.model_df[arg["VARIABLE"]].as_matrix()
        narray = narray.astype(np.float64)
        scaler = StandardScaler()
        nscaled = scaler.fit_transform(narray)
        nscaled = nscaled * 0.1
        self.model_df[arg["VARIABLE"]] = nscaled

    def create_train_test(self):
        train = self.model_df.loc[self.model_df["CLASS"] == "TRAIN"]
        test = self.model_df.loc[self.model_df["CLASS"] == "TEST"]
        self.log.info("Training on %d molecules", train.shape[0])
        self.log.info("Testing (fitting) on %d molecules", test.shape[0])
        if train.shape[0] == 0 or test.shape[0] == 0:
            self.log.error('Zero records for training and/or testing, check "CLASS" var is either "TEST" or "TRAIN"')
            sys.exit()
        return train, test



class AccessData(object): # The adapter class actually used to return data to the model.

    def __init__(self, model_data_obj, data):
        self.log = model_data_obj.log
        self.args = model_data_obj.args
        self.model_args = model_data_obj.model_args
        self.data = data

    def get_field(self, var):
        return self.data[var].tolist()

    def target_data(self):
        return self.data[self.model_args["DEPENDENT"]["VARIABLE"]].values # return as a numpy array

    def input_data(self):
        if len(self.model_args["INDEPENDENT"]) == 1:

            return self.get_tensor(self.model_args["INDEPENDENT"][0]["VARIABLE"])

        elif len(self.model_args["INDEPENDENT"]) > 1:

            tensor_dict = {}
            for var in self.model_args["INDEPENDENT"]:
                tensor = self.get_tensor(var["VARIABLE"])
                tensor_dict[var["VARIABLE"]] = tensor

            return tensor_dict   # return as a dictionary

        else: # zero independent arguments - error
            self.log.error("No independent variables defined for model.")
            sys.exit()

    def get_tensor(self, var_name):
        # convenience routine to convert a pandas column to a tensor
        data = self.data[var_name]
        data_list = data.tolist()
        if len(data_list) == 0:
            self.log.error("X (input) Variable: %s has zero rows", var_name)
            sys.exit()
        data_shape = data_list[0].shape
        array_shape = (len(data_list),) + data_shape
        tensor = np.ndarray(shape=array_shape)
        for i, x in enumerate(data_list):
            tensor[i] = x
        return tensor
