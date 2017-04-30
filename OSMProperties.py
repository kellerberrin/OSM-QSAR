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
import os
import numpy as np
import pandas as pd
from math import log10

from sklearn.decomposition import PCA, KernelPCA

from rdkit import Chem
from rdkit.Chem import AllChem

import deepchem as dc
from deepchem.feat import Featurizer, CoulombMatrix, CoulombMatrixEig


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
        self.dragon_fields = self.read_dragon_fields("DragonFields.csv")
        self.dragon_truncation_rank = 100
        self.trunc_dragon = self.truncated_dragon()
        self.max_atoms = self.check_smiles(self.data)

        if self.args.coulombFlag:
            self.generate_coulomb_matrices()

        self.generate_fields()

        if self.args.varFlag: # If the "--vars" flag is specified then list the variables and exit.
            self.log.info("The Available Modelling Variables are:")
            self.display_variables()
            sys.exit()


    def display_variables(self):
            for column in self.data.columns:
                msg = "Variable: {:20s} Dimension: {}".format(column, self.get_dimension(column))
                self.log.info(msg)

    def get_dragon_headers(self):
        # Note this returns 1666 columns in the same order that the columns are in the pandas array.
        # The first column is a SMILE and is removed.
        field_list = list(self.dragon.columns.values)
        field_list.pop(0)
        return field_list

    def get_dragon_fields(self):
        # Sorts field info into dragon header order and returns pandas data frame.
        sorted_dragon_fields = pd.DataFrame(pd.Series(self.get_dragon_headers()), columns=["FIELD"])
        sorted_dragon_fields = pd.merge(sorted_dragon_fields, self.dragon_fields, how="inner", on=["FIELD"])
        sorted_dragon_fields = sorted_dragon_fields.drop_duplicates(subset=["FIELD"], keep="first")
        after_merge = list(sorted_dragon_fields["FIELD"])
        missing_list = list(set(self.get_dragon_headers()) - set(after_merge))
        for missing in missing_list:
            self.log.error("Dropped FIELD ID: %s after sort and merge with Dragon headers", missing)
        if len(missing_list) > 0:
            sys.exit()
        return sorted_dragon_fields

    def get_truncated_dragon_headers(self):
        return self.trunc_dragon

    def get_truncated_dragon_fields(self):
        return self.dragon_fields.loc[self.dragon_fields["RANK"] <= self.dragon_truncation_rank]

    def get_data(self):
        return self.data

    def get_dimension(self, column_name):
        if self.data[column_name].shape[0] == 0:
            self.log.error("Unexpected Error column %s contains no rows", column_name)
            sys.exit()
        if isinstance(self.data[column_name][0], np.ndarray):
            shape = self.data[column_name][0].shape
            return shape
        else:
            return 1

    def generate_fields(self):

        self.log.info("Calculating QSAR fields, may take a few moments....")
        # Add the finger print columns
        self.generate_fingerprints()
        # Add the potency classes
        self.generate_potency_class(200)
        self.generate_potency_class(500)
        self.generate_potency_class(1000)
        # Add pEC50
        self.generate_pEC50()
        # ION_ACTIVE class reduces ION_ACTIVITY to 2 classes ["ACTIVE", "INACTIVE"]
        self.generate_ion_active()


    def generate_coulomb_matrices(self):

        self.log.info("Generating Coulomb Matrices, may take a few moments ...")

        matrix_featurizer = CoulombMatrix(self.max_atoms, randomize=False, n_samples=1)
        eigen_featurizer = CoulombMatrixEig(self.max_atoms)

        matrices = []
        smiles = []
        arrays = []
        eigenarrays = []
        num_confs = 1

        for index, row in self.data.iterrows():
            mol = Chem.MolFromSmiles(row["SMILE"])
            Chem.AddHs(mol)
            ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs)
            if len(ids) != num_confs:
                ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, ignoreSmoothingFailures=True)
                if len(ids) != num_confs:
                    self.log.warning("Coulomb Matrix - unable to generate %d conformer(s) for smile: %s"
                                     , num_confs, row["SMILE"])

            if len(ids) == num_confs:
                for id in ids:
                    AllChem.UFFOptimizeMolecule(mol, confId=id)
                matrix = matrix_featurizer.coulomb_matrix(mol)
                matrices.append(matrix)
                arrays.append(matrix[0].flatten())
                smiles.append(row["SMILE"])
                eigenvalues = eigen_featurizer.featurize([mol])
                eigenarrays.append(eigenvalues[0].flatten())

        pd_dict = { "SMILE": smiles, "COULOMB": matrices, "COULOMB_ARRAY": arrays, "COULOMB_EIGEN" : eigenarrays }
        coulomb_frame = pd.DataFrame(pd_dict)

        before_ids = list(self.data["ID"])
        self.data = pd.merge(self.data, coulomb_frame, how="inner", on=["SMILE"])
        self.data = self.data.drop_duplicates(subset=["SMILE"], keep="first")
        after_ids = list(self.data["ID"])
        missing_list = list(set(before_ids) - set(after_ids))
        for missing in missing_list:
            self.log.warning("Dropped molecule ID: %s after join with Coulomb Matrix data", missing)

    def generate_potency_class(self, nMol):

        column_name = "EC50_{}".format(int(nMol))
        EC50 = self.data["EC50"]
        potency_class = []
        for ec50 in EC50:
            if pd.isnull(ec50) or ec50 <= 0:
                potency_class.append(ec50)
            else:
                klass = "ACTIVE" if ec50 * 1000 <= nMol else "INACTIVE"
                potency_class.append(klass)
        self.data[column_name] = pd.Series(potency_class, index=self.data.index)

    def generate_pEC50(self):

        column_name = "pEC50"
        EC50 = self.data["EC50"]
        pEC50_list = []
        for ec50 in EC50:
            if pd.isnull(ec50) or ec50 <= 0:
                pEC50_list.append(ec50)
            else:
                pEC50_list.append(log10(ec50))
        self.data[column_name] = pd.Series(pEC50_list, index=self.data.index)


    def generate_ion_active(self):

        column_name = "ION_ACTIVE"
        ion_activity = self.data["ION_ACTIVITY"]
        ion_active = []
        for ion in ion_activity:
            if pd.isnull(ion):
                ion_active.append(np.NaN)
            else:
                klass = "ACTIVE" if ion == "ACTIVE" else "INACTIVE"
                ion_active.append(klass)
        self.data[column_name] = pd.Series(ion_active, index=self.data.index)

    def generate_fingerprints(self):

        # Add the finger print columns
        morgan_1024 = lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 4, nBits=1024)
        morgan_2048_1 = lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 1, nBits=2048)
        morgan_2048_2 = lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048)
        morgan_2048_3 = lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 3, nBits=2048)
        morgan_2048_4 = lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 4, nBits=2048)
        morgan_2048_5 = lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 5, nBits=2048)
        morgan_2048_6 = lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 6, nBits=2048)
        topological_2048 = lambda x: AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(x)
        macc = lambda x: AllChem.GetMACCSKeysFingerprint(x)

        self.add_bitvect_fingerprint(self.data, morgan_1024, "MORGAN1024")
        self.add_bitvect_fingerprint(self.data, morgan_2048_1, "MORGAN2048_1")
        self.add_bitvect_fingerprint(self.data, morgan_2048_2, "MORGAN2048_2")
        self.add_bitvect_fingerprint(self.data, morgan_2048_3, "MORGAN2048_3")
        self.add_bitvect_fingerprint(self.data, morgan_2048_4, "MORGAN2048_4")
        self.add_bitvect_fingerprint(self.data, morgan_2048_5, "MORGAN2048_5")
        self.add_bitvect_fingerprint(self.data, morgan_2048_6, "MORGAN2048_6")
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

        except IOError:
            self.log.error('Problem reading EDragon file %s, Check "--data", ""--dir" and --help" flags.', file_name)
            sys.exit()

        self.log.info("Read %d records from file %s", data_frame.shape[0], file_name)

        return dragon_data_frame


    def read_dragon_fields(self, file_name):

        path_name = os.path.join(self.args.workDirectory, file_name)

        self.log.info("Loading EDragon Ranked Fields: %s ...", path_name)

        try:

            dragon_field_frame = pd.read_csv(path_name, low_memory=False)

        except IOError:
            self.log.error('Problem reading EDragon Field file %s', path_name)
            sys.exit()

        self.log.info("Read %d records from file %s", dragon_field_frame.shape[0], path_name)

        return dragon_field_frame

    def truncated_dragon(self):

        trunc_fields = self.get_truncated_dragon_fields()
        field_list = trunc_fields["FIELD"].tolist()
        new_frame = pd.DataFrame(self.dragon, columns=["SMILE"])
        data_frame = pd.DataFrame(self.dragon, columns=field_list)
        narray = data_frame.as_matrix(columns=None)
        narray = narray.astype(np.float64)
        narray_list = [ x for x in narray]
        new_frame["TRUNC_DRAGON"] = pd.Series(narray_list, index=data_frame.index)
        before_ids = list(self.data["ID"])
        self.data = pd.merge(self.data, new_frame, how="inner", on=["SMILE"])
        self.data = self.data.drop_duplicates(subset=["SMILE"], keep="first")
        after_ids = list(self.data["ID"])
        missing_list = list(set(before_ids) - set(after_ids))

        for missing in missing_list:
            self.log.warning("Dropped molecule ID: %s after join with TRUNC_DRAGON data", missing)

        return field_list

    # Read CSV File into a pandas data frame
    def read_csv(self, file_name):

        self.log.info("Loading data file: %s ...", file_name)
        mandatory_fields = ["EC50", "SMILE", "ID", "CLASS", "ION_ACTIVITY"]

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
        # also calculates the maximum number of atoms for calculating
        # coulomb matrices

        max_atoms = 0

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

            num_atoms = mol.GetNumAtoms()
            if num_atoms > max_atoms:
                max_atoms = num_atoms

        self.log.info("Maximum Molecular Atoms: %d", max_atoms)

        return max_atoms
