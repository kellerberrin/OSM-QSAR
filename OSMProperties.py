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
        morgan_2048_3 = lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 3, nBits=2048)
        morgan_2048_4 = lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 4, nBits=2048)
        morgan_2048_5 = lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 5, nBits=2048)
        topological_2048 = lambda x: AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(x)
        macc = lambda x: AllChem.GetMACCSKeysFingerprint(x)

        self.add_bitvect_fingerprint(self.data, morgan_1024, "MORGAN1024")
        self.add_bitvect_fingerprint(self.data, morgan_2048, "MORGAN2048")
        self.add_bitvect_fingerprint(self.data, morgan_2048_3, "MORGAN2048_3")
        self.add_bitvect_fingerprint(self.data, morgan_2048_4, "MORGAN2048_4")
        self.add_bitvect_fingerprint(self.data, morgan_2048_4, "MORGAN2048_5")
        self.add_bitvect_fingerprint(self.data, topological_2048, "TOPOLOGICAL2048")
        self.add_bitvect_fingerprint(self.data, macc, "MACCFP")

        if self.args.varFlag: # If the "--vars" flag is specified then list the variables and exit.
            self.log.info("The Available Modelling Variables are:")
            self.display_variables()
            sys.exit()

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
                sys.exit()

        except IOError:
            self.log.error('Problem reading data file %s, Check the "--data", ""--dir" and --help" flags.', file_name)
            sys.exit()

        self.log.info("Read %d records from file %s", data_frame.shape[0], file_name)

        return data_frame

    # Generate the molecular fingerprints..
    def add_bitvect_fingerprint(self, data_frame, finger_printer, column_name):
        """ Generate molecular fingerprints as a numpy array of floats"""

        int_list = []
        for index, row in data_frame.iterrows():

            mol = Chem.MolFromSmiles(row["SMILE"])
            fp = finger_printer(mol)
            int_fp = [int(x) for x in fp]
            np_fp = np.array(int_fp, dtype=float)
            int_list.append(np_fp)
        # Store a list of numpy.arrays
        data_frame[column_name] = pd.Series(int_list, index=data_frame.index)

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

