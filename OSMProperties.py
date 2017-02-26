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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#
#
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import numpy

from rdkit import Chem
from rdkit.Chem import AllChem

# ===================================================================================================
#
# This class reads in the training/test CSV file (no header).
# The CSV fields are - EC50 (in uMol), TestID, SMILES, CompoundID 
#
# Thee classes also generates molecular properties from the ligand SMILES.
# The property currently generated is GetMorganFingerprint.
# See the rdkit documentation for further info.
# Further properties are easily added by returning a dictionary
# and concatonating this to the class member dictionaries using "merge()".
# e.g. "self.merge(self.train,self.morgan(train["SMILE"),self.otherpropertyA(train[..]),...etc)".
#
# Currently returns two dictionaries: "train" (training data) and "test" (classification data)
# These dictionaries are member variables in the "Properties" class and can be
# accessed once the object has been created as: "PropertiesObj.train" and "PropertiesObj.test"
# The dictionaries contain the following entries, each entry is an array of all compounds:
# "pEC50" a float list of log10(EC50) values for each compound.
# "CLASS" a string list tagging the entry as "TRAIN" or "TEST".
# "SMILE" a string list of the molecular smiles for each compound.
# "ID" a string list the OSM ids given to the compounds.
# "MORGAN" a float list of the Morgan fingerprint of each compound.
# ===================================================================================================


class Properties(object):
    """Generate molecular properties from SMILES"""

# Concatonate an arbitrary number of dictionaries to create the train and test dictionaries.
# Currently only Morgan fingerprints are generated. .
    def __init__(self, args, log):

# Shallow copy of the environment variables. 
        self.log = log
        self.args = args
# Retrieve the data from CSV file.
        self.train, self.test = self.read_csv(self.args.dataFilename)

# Create the properties training dictionary.
        self.train = self.merge(self.train,
                                self.morgan1024(self.train["SMILE"]),
                                self.morgan2048(self.train["SMILE"]))

# Create the properties test dictionary.
        self.test = self.merge(self.test,
                               self.morgan1024(self.test["SMILE"]),
                               self.morgan2048(self.test["SMILE"]))

# Read the CSV file.    
    def read_csv(self, file_name):

        self.log.info("Loading data file: %s ...", file_name)

        file_data = []
        found_header = False
        header_line = ""
        mandatory_fields = ["pEC50", "SMILE", "ID", "CLASS"]

        try:

            with open(file_name, 'r') as in_file:
                content = in_file.readlines()

            content = [x.strip() for x in content]

            for line in content:   # Strip comments and setup header.
                if line[:1] != "#":   # The first character of a comment line must be a "#".
                    if not found_header:
                        found_header = True
                        header_line = line
                    else:
                        file_data.append(line)

            header_list = [x.strip() for x in header_line.split(",")]
            self.log.info("%s contains %d data fields", file_name, len(header_list))

            if not set(mandatory_fields) <= set(header_list):
                self.log.error("Mandatory data fields %s absent.", ",".join(mandatory_fields))
                self.log.error("File %s contains fields %s.", file_name, ",".join(header_list))
                self.log.fatal("OSM_QSAR cannot continue.")
                sys.exit()

            train_data = [[] for x in range(len(header_list))]
            test_data = [[] for x in range(len(header_list))]
            class_index = header_list.index("CLASS")
            pEC50_index = header_list.index("pEC50")

            line_no = 1
            for line in file_data:
                field_list = [x.strip() for x in line.split(",")]

                if len(field_list) != len(header_list):
                    self.log.error("Incorrect number of data fields: %d, expected:%d at data_line:%d",
                                   len(field_list), len(header_list), line_no)
                    self.log.error("File %s contains fields %s.", file_name, ",".join(header_list))
                    self.log.fatal("OSM_QSAR cannot continue.")
                    sys.exit()

                field_list[pEC50_index] = float(field_list[pEC50_index])

                if field_list[class_index].upper() == "TEST":
                    for idx in range(len(field_list)):
                        test_data[idx].append(field_list[idx])
                else:
                    for idx in range(len(field_list)):
                        train_data[idx].append(field_list[idx])

                line_no += 1

            train_dict = {}
            test_dict = {}
            for idx in range(len(header_list)):
                train_dict[header_list[idx]] = train_data[idx]
                test_dict[header_list[idx]] = test_data[idx]

            self.log.info("%s contains a total of %d training molecules", file_name, len(train_data[0]))
            self.log.info("%s contains a total of %d test molecules", file_name, len(test_data[0]))

        except IOError:
            self.log.error('Problem reading data file %s, Check the "--data", ""--dir" and --help" flags.', file_name)
            self.log.fatal("OSM_QSAR cannot continue.")
            sys.exit()

        return train_dict, test_dict

# Utility function to merge property dictionaries.
    def merge(self, *dict_args):
        """ Given any number of dicts, shallow copy and merge into a new dict."""
    
        result = {}
        for dictionary in dict_args:
            result.update(dictionary)
        return result


# Generate the Morgan molecular fingerprint for 1024 bits..
    def morgan1024(self, smiles):
        """ Generate Morgan molecular properties as a dictionary containing an numpy array of float[1024]"""

        mols = [Chem.MolFromSmiles(x) for x in smiles]
        bit_info = {}
        morgan_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 4, nBits=1024, bitInfo=bit_info) for x in mols]

        int_list = []
        for arr in morgan_fps:
            int_list.append([int(x) for x in arr])

        morgan_floats = numpy.array(int_list, dtype=float)
        return { "MORGAN1024" : morgan_floats }

# Generate the Morgan molecular fingerprint for 2048 bits.
    def morgan2048(self, smiles):
        """ Generate Morgan molecular properties as a dictionary containing an numpy array of float[2048]"""

        mols = [Chem.MolFromSmiles(x) for x in smiles]
        bit_info = {}
        morgan_fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048, bitInfo=bit_info) for x in mols]

        int_list = []
        for arr in morgan_fps:
            int_list.append([int(x) for x in arr])

        morgan_floats = numpy.array(int_list, dtype=float)
        return { "MORGAN2048" : morgan_floats }


