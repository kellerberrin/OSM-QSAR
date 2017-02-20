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
import csv
import math
import operator
import numpy
import argparse
import logging

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# ===================================================================================================
#
# This class reads in the training/test CSV file (no header).
# The CSV fields are - EC50 (in uMol), TestID, SMILES, CompoundID 
#
# Thee classes also generates molecular properties from the ligand SMILES.
# The property currently generated is GetMorganFingerprint.
# See the rdkit documentation for further info.
# Futher properties are easily added by returning a dictionary
# and concatonating this to the class member dictionaries using "merge()".
# e.g. "self.merge(self.common(),self.morgan(),self.otherpropertyA(),self.otherpropertyB(),...etc)".   
#
# Currently returns two dictionaries: "train" (training data) and "test" (classification data)
# These dictionaries are member variables in the "Properties" class and can be
# accessed once the object has been created as: "PropertiesObj.train" and "PropertiesObj.test"
# The dictionaries currently contain the following entries, each entry is an array of all compounds:
# "pEC50" an array of log10(EC50) values for each compound.
# "assay" an array of the assay classification from the original CSV data.
# "smiles" an array the molecular smiles for each compound. 
# "ids" an array the OSM ids given to the compounds.
# "morgan" an array of the Morgan fingerprint of each compound implemented as an array of floats. 
# ===================================================================================================



class Properties(object):
    """Generate molecular properties from SMILES"""

# Concatonate an arbitrary number of dictionaries to create the train and test dictionaries.
# Currently only Morgan fingerprints are generated and concatonated.
    def __init__(self, args, log):

# Shallow copy of the environment variables. 
        self.log = log
        self.args = args
# Retrieve the data from CSV file.
        self.fileData = self.read_csv(self.args.dataFilename)
# Split the data for Training (Lit, M, and A categories) from testing 
        self.splitData = self.split_types(self.fileData)
# Create the properties training dictionary.
        self.train = self.merge(self.common(self.splitData["train"]), self.morgan(self.splitData["train"]))
# Create the properties test dictionary.
        self.test = self.merge(self.common(self.splitData["test"]), self.morgan(self.splitData["test"]))

# Read the CSV file.    
    def read_csv(self, fileName):

        self.log.info("Loading data file: %s ...", fileName)

        fileData = []

        with open(fileName, 'r') as inFile:
            content = csv.reader(inFile, delimiter=',')

            for row in content:
                fileData.append(row)

#        print fileData[0]

        return fileData

# Split into train and test data.
    def split_types(self, fileData):

        trainSets = ['M', 'A', 'Lit']
        trainData = []
        testData = []
        for x in fileData:
            if x[2] in trainSets:
                trainData.append(x)
            else:
                testData.append(x)

        self.log.info("%s contains a total of %d training molecules", self.args.dataFilename, len(trainData))
        self.log.info("%s contains a total of %d test molecules", self.args.dataFilename, len(testData))

        # Return a dictionary with the split data.

        return { "train" : trainData, "test" : testData}
    
# Utility function to merge property dictionarys.    
    def merge(self, *dict_args):
        """ Given any number of dicts, shallow copy and merge into a new dict."""
    
        result = {}
        for dictionary in dict_args:
            result.update(dictionary)
        return result

# Generate the common properties.    
    def common(self, data):
        """ Generate common tagged properties."""

        potency = [math.log10(float(x[0])) for x in data]
        assay_classification = [x[1] for x in data]
        molset = [x[2] for x in data]
        smiles = [x[3] for x in data]
        ids = [x[4] for x in data]

        return { "pEC50" : potency, "assay" : assay_classification, "smiles" : smiles, "ids" : ids }

# Generate the Morgan molecular property.
    def morgan(self, data):
        """ Generate Morgan molecular properties as a dictionary containing an array of float[1024]"""

        smiles = [x[3] for x in data]
        mols = [Chem.MolFromSmiles(x) for x in smiles]
        bitInfo = {}
        morganBits = [AllChem.GetMorganFingerprintAsBitVect(x, 4, nBits=1024, bitInfo=bitInfo) for x in mols]

        intArray = []
        for arr in morganBits:
            intArray.append([int(x) for x in arr])

        morganFloats = numpy.array(intArray, dtype=float)

        return { "morgan" : morganFloats }



