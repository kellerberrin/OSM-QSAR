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

import sys
import numpy as np
import pandas as pd
import copy


from OSMProperties import OSMGenerateData

# ===================================================================================================
#
# Presents the 'model-centric' view of the data.
#
# ===================================================================================================

class OSMModelData(object):

    def __init__(self, args, log, model, data):

        self.log = log
        self.args = args
        self.model_args = model.model_arguments()
        self.model_df = copy.deepcopy(data.get_data())    # The data object may used elsewhere, leave it unmolested.
        self.train, self.test = self.setup_model_data()

    def training(self):  # Access training data (returns the adapter class).
        return AccessData(self.args, self.log, self.train, self.model_args)

    def testing(self):   # Access the test (target) data (returns the adapter class).
        return AccessData(self.args, self.log, self.test, self.model_args)

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
            if len(self.args.indepList) != len(self.model_args["INDEPENDENT"]):
                self.log.error("Model requires %d independent variables;  %s specified", ",".join(self.args.indepVar))
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

        self.check_shape(arg)

    def check_shape(self, arg):

        arg_shape =arg["SHAPE"]

        if arg_shape is None: return    # no shape checking

        df_shape = self.model_df[arg["VARIABLE"]].shape

        if arg_shape[0] >= 0:
            if arg_shape[0] != df_shape[0]:
                self.log.error("Variable %s requires %d rows, data frame has %d."
                               , arg["VARIABLE"], arg_shape[0], df_shape[0])
                sys.exit()

        if self.model_df[arg["VARIABLE"]].shape[0] == 0:
            self.log.error("No valid rows for variable %s", arg["VARIABLE"])
            sys.exit()


        if len(arg_shape) > 1:          # note we can only store lists of numpys in pd.data_frames (why?)

                # Get the shape of the first data item (assumed to be a numpy.array).
            var_shape = self.model_df[arg["VARIABLE"]][0].shape

            for idx in range(len(arg_shape)-1):

                if arg_shape[idx+1] < 0:
                    continue
                else:
                    if arg_shape[idx+1] != var_shape[idx]:
                        self.log.error("Variable %s[%d] has shape %d, model requires %d.", arg["VARIABLE"], idx+1
                                        , var_shape[idx], arg_shape[idx+1])
                        sys.exit()

        elif isinstance(self.model_df[arg["VARIABLE"]][0], np.ndarray):  # Unexpected numpy.
            self.log.error("Expecting scalar variable %s, got numpy array.", arg["VARIABLE"])
            sys.exit()

    def convert_type(self, arg):

        if not isinstance(self.model_df[arg["VARIABLE"]][0], np.ndarray):  # only if scalar otherwise a numpy array.

            if arg["TYPE"] == np.float64: # convert to numeric.
                try:
                    pd.to_numeric(self.model_df[arg["VARIABLE"]])
                except ValueError:
                    self.log.error("Problem converting variable %s to floats", arg["VARIABLE"])
                    sys.exit()
            elif arg["TYPE"] == np.str:
                self.model_df[arg["VARIABLE"]].replace("", np.nan, inplace=True) # Convert empty fields to NaNs
            else:
                self.model_df[arg["VARIABLE"]].replace("", np.nan, inplace=True) # Convert empty fields to NaNs

            self.model_df.dropna(subset=[arg["VARIABLE"]], inplace=True) # Delete all NaN rows.

        if self.model_df[arg["VARIABLE"]].shape[0] == 0:
            self.log.error("No valid values in for variable %s (check string or numeric)", arg["VARIABLE"])
            sys.exit()

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

    def __init__(self, args, log, data, model_args):
        self.log = log
        self.args = args
        self.data = data
        self.model_args = model_args

    def get_field(self, var):
        return self.data[var].tolist()

    def target_data(self):
        return self.data[self.model_args["DEPENDENT"]["VARIABLE"]].values # return as a numpy array

    def input_data(self):
        if len(self.model_args["INDEPENDENT"]) == 1:
            matrix = np.matrix(self.data[self.model_args["INDEPENDENT"][0]["VARIABLE"]].tolist())
            return matrix # numpy matrix.
        elif len(self.model_args["INDEPENDENT"]) > 1:
            matrix_list = []
            for var in self.data["INDEPENDENT"]:
                matrix = self.data[var["VARIABLE"].tolist()]
                matrix_list.append(matrix)
            return matrix_list   # return as a list of numpy matrices
        else: # zero independent arguments - error
            self.log.error("No independent variables defined for model.")
            sys.exit()

