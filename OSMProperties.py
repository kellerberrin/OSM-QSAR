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

from rdkit import Chem
from rdkit.Chem import AllChem

from OSMBase import OSMBaseModel

# ===================================================================================================
#
# This class reads in the training/test CSV file.
# The class also generates molecular properties from the SMILES.
# See the rdkit documentation for further info.
#
# ===================================================================================================

class OSMGenerateData(object):
    """Generate molecular properties"""

    def __init__(self, args, log):

        self.log = log
        self.args = args
        self.data = self.read_csv(self.args.dataFilename)

        self.check_smiles(self.data)

        # Add the finger print columns
        morgan_1024 = lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 4, nBits=1024)
        morgan_2048 = lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048)

        self.add_fingerprint(self.data, morgan_1024, "MORGAN1024")
        self.add_fingerprint(self.data, morgan_2048, "MORGAN2048")

    def display_variables(self):
            for column in self.data.columns:
                self.log.info("Variable: %s", column)

    def get_data(self):
        return self.data

    # Read CSV File into a pandas data frame
    def read_csv(self, file_name):

        self.log.info("Loading data file: %s ...", file_name)
        mandatory_fields = ["pIC50", "SMILE", "ID", "CLASS"]

        try:

            data_frame = pd.read_csv(file_name)

            if not set(mandatory_fields) <= set(data_frame):
                self.log.error("Mandatory data fields %s absent.", ",".join(mandatory_fields))
                self.log.error("File %s contains fields %s.", file_name, ",".join(data_frame))
                self.log.fatal("OSM_QSAR cannot continue.")
                sys.exit()

        except IOError:
            self.log.error('Problem reading data file %s, Check the "--data", ""--dir" and --help" flags.', file_name)
            self.log.fatal("OSM_QSAR cannot continue.")
            sys.exit()

        self.log.info("Read %d records from file %s", data_frame.shape[0], file_name)

        return data_frame

    # Generate the molecular fingerprints..
    def add_fingerprint(self, data_frame, finger_printer, column_name):
        """ Generate molecular fingerprints as a numpy array of floats"""

        int_list = []
        for index, row in data_frame.iterrows():

            mol = Chem.MolFromSmiles(row["SMILE"])
            fp = finger_printer(mol)
            int_fp = [int(x) for x in fp]
            int_list.append(int_fp)

        np_fp = np.array(int_list, dtype=float)
        data_frame[column_name] = pd.Series(np_fp.tolist(), index=data_frame.index)

    def check_smiles(self, data_frame):
        # Check all the "SMILES" and ensure they are valid.

        for index, row in data_frame.iterrows():

            mol = Chem.MolFromSmiles(row["SMILE"])

            try:
                result = Chem.SanitizeMol(mol)
                if result != Chem.SanitizeFlags.SANITIZE_NONE:
                    sanitized_smile = Chem.MolToSmiles(mol)
                    self.log.warning("Sanitized SMILE %s, Compound ID:%s", sanitized_smile, data_frame[index, "ID"])
                    data_frame.set_value(index, "SMILE", sanitized_smile)
            except:
                self.log.warning("Unable to Sanitize SMILE %s, Compound ID:%s", row["SMILE"] , row["ID"])
                self.log.warning("Record Deleted. OSM_QSAR attempts to continue ....")
                data_frame.drop(index, inplace=True)


class AccessData(object): # The facade class actually used by the model.

    def __init__(self, args, log, data, depend_args, indep_args):
        self.log = log
        self.args = args
        self.data = data
        self.dependent_var = depend_args
        self.independent_var_list = indep_args

    def get_field(self, var):
        return self.data[var].tolist()

    def target_data(self):
        return self.data[self.dependent_var].values # return as a numpy array

    def input_data(self):
        matrix_list = []
        for var in self.independent_var_list:
            matrix = np.array(self.data[var].tolist(), dtype = float)
            matrix_list.append(matrix)
        return matrix_list   # return as a list of numpy matrices

class OSMModelData(object):

    def __init__(self, args, log, model, data):

        self.log = log
        self.args = args
        self.data = data.get_data()
        self.dependent_var = model.model_arguments()["DEPENDENT"]
        self.independent_var_list = model.model_arguments()["INDEPENDENT"]
        self.train, self.test = self.setup_model_data(model)

    def training(self):
        return AccessData(self.args, self.log, self.train, self.dependent_var, self.independent_var_list)

    def testing(self):
        return AccessData(self.args, self.log, self.test, self.dependent_var, self.independent_var_list)

########################################################################################################
# Local member functions.
########################################################################################################

    def setup_model_data(self, model):
        self.setup_dependent_variable(model)
        self.setup_independent_variables(model)
        return self.create_train_test(model)

    def setup_dependent_variable(self, model):

        if self.dependent_var not in self.data.columns:
            self.args.log.error('Model dependent variable %d not found in data frame, check with "--vars"',
                                self.dependent_var)
            self.log.fatal("OSM_QSAR cannot continue.")
            sys.exit()

        if model.model_is_regression(): # convert to numeric.
            try:
                pd.to_numeric(self.data[self.dependent_var])
            except ValueError:
                self.log.error("Problem converting dependent variable %s to floats", self.dependent_var)
                self.log.fatal("OSM_QSAR cannot continue.")
                sys.exit()
        else: # the model is a classifier
            self.data[self.dependent_var].replace("", np.nan, inplace=True) # Convert empty fields to NaNs

        self.data.dropna(subset=[self.dependent_var], inplace=True) # Delete all NaN rows.

        if self.data.shape[0] == 0:
            self.log.error("No valid values in for dependent variable %s (check string or numeric)", self.dependent_var)
            self.log.fatal("OSM_QSAR cannot continue.")
            sys.exit()

    def setup_independent_variables(self, model):

        if not set(self.independent_var_list) <= set(self.data.columns):
            self.args.log.error('Model %s independent variables %s not found in data frame, check with "--vars"'
                               , model.model_name(), ",".join(self.independent_var_list))
            self.log.fatal("OSM_QSAR cannot continue.")
            sys.exit()

    def create_train_test(self, model):
        train = self.data.loc[self.data["CLASS"] == "TRAIN"]
        test = self.data.loc[self.data["CLASS"] == "TEST"]
        self.log.info("Model %s training on %d molecules", model.model_name(), train.shape[0])
        self.log.info("Model %s testing (fitting) on %d molecules", model.model_name(), test.shape[0])
        return train, test

