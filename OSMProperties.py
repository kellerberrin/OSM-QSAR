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

from sklearn.decomposition import PCA, KernelPCA

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
        self.dragon = self.read_dragon(self.args.dragonFilename)

        self.check_smiles(self.data)

        self.generate_fields()

        if self.args.varFlag: # If the "--vars" flag is specified then list the variables and exit.
            self.log.info("The Available Modelling Variables are:")
            self.display_variables()
            sys.exit()


    def display_variables(self):
            for column in self.data.columns:
                msg = "Variable: {:20s} Dimension: {:4d}".format(column, self.get_dimension(column))
                self.log.info(msg)

    def get_dragon_headers(self):
        return list(self.dragon.columns.values)

    def get_data(self):
        return self.data

    def get_dimension(self, column_name):
        if self.data[column_name].shape[0] == 0:
            self.log.error("Unexpected Error column %s contains no rows", column_name)
            sys.exit()
        if isinstance(self.data[column_name][0], np.ndarray):
            return self.data[column_name][0].shape[0]
        else:
            return 1

    def generate_fields(self):

        self.log.info("Calculating QSAR fields, may take a few moments....")

        # Add the finger print columns
        morgan_1024 = lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 4, nBits=1024)
        morgan_2048_1 = lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 1, nBits=2048)
        morgan_2048_2 = lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048)
        morgan_2048_3 = lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 3, nBits=2048)
        morgan_2048_4 = lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 4, nBits=2048)
        morgan_2048_5 = lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 5, nBits=2048)
        topological_2048 = lambda x: AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(x)
        macc = lambda x: AllChem.GetMACCSKeysFingerprint(x)

        self.add_bitvect_fingerprint(self.data, morgan_1024, "MORGAN1024")
        self.add_bitvect_fingerprint(self.data, morgan_2048_1, "MORGAN2048_1")
        self.add_bitvect_fingerprint(self.data, morgan_2048_2, "MORGAN2048_2")
        self.add_bitvect_fingerprint(self.data, morgan_2048_3, "MORGAN2048_3")
        self.add_bitvect_fingerprint(self.data, morgan_2048_4, "MORGAN2048_4")
        self.add_bitvect_fingerprint(self.data, morgan_2048_5, "MORGAN2048_5")
        self.add_bitvect_fingerprint(self.data, topological_2048, "TOPOLOGICAL2048")
        self.add_bitvect_fingerprint(self.data, macc, "MACCFP")


    # Read the Dragon data as a pandas object with 2 fields, [SMILE, np.ndarray] and join (merge)
    # on "SMILE" with the OSMData data.
    def read_dragon(self, file_name):

        self.log.info("Loading EDragon QSAR file: %s ...", file_name)

        try:

            dragon_data_frame = pd.read_csv(file_name, low_memory=False)
            new_frame = pd.DataFrame(dragon_data_frame, columns=["SMILE"])
            data_frame = dragon_data_frame.drop("SMILE", 1)
            narray = data_frame.as_matrix(columns=None)
            narray = narray.astype(np.float64)
            narray_list = [ x for x in narray]
            new_frame["DRAGON"] = pd.Series(narray_list, index=data_frame.index)
            before_ids = list(self.data["ID"])
            self.data = pd.merge(self.data, new_frame, how="inner", on=["SMILE"])
            self.data = self.data.drop_duplicates(subset=["SMILE"], keep="first")
            after_ids = list(self.data["ID"])
            missing_list = list(set(before_ids) - set(after_ids))
            for missing in missing_list:
                self.log.warning("Dropped molecule ID: %s after join with DRAGON data", missing)

            pca = PCA(n_components=10)
            pca.fit(narray_list)
            pca_array = pca.transform(narray_list)
            pca_list = [ x for x in pca_array]
#            self.data["DRAGON_PCA"] = pd.Series(pca_list, index=data_frame.index)

            kpca = KernelPCA(n_components=10, kernel="rbf")
            kpca.fit(narray_list)
            kpca_array = kpca.transform(narray_list)
            kpca_list = [ x for x in kpca_array]

        except IOError:
            self.log.error('Problem reading EDragon file %s, Check "--data", ""--dir" and --help" flags.', file_name)
            sys.exit()

        self.log.info("Read %d records from file %s", data_frame.shape[0], file_name)

        return dragon_data_frame

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
                    self.log.warning("Sanitized SMILE %s, Compound ID:%s", sanitized_smile, row["ID"])
                    data_frame.set_value(index, "SMILE", sanitized_smile)
            except:
                self.log.warning("Unable to Sanitize SMILE %s, Compound ID:%s", row["SMILE"] , row["ID"])
                self.log.warning("Record Deleted. OSM_QSAR attempts to continue ....")
                data_frame.drop(index, inplace=True)

