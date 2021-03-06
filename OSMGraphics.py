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

import os
import sys

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps


# ===============================================================================
# Graphics object currently only implements Gregory Landrum's Similarity Maps.
# ===============================================================================

class OSMSimilarityMap(object):

    def __init__(self, model, data, func):

        self.args = model.args
        self.log = model.log
        self.model = model
        self.data = data
        self.fingerprinter = None
        self.probability_func = None

        self.fingerprinter = self.check_args()
        self.probability_func = func

    def check_args(self):

        model_args = self.model.model_arguments()
        if len(model_args["INDEPENDENT"]) == 1:             # Only one fingerprint

            var_name = model_args["INDEPENDENT"][0]["VARIABLE"]

            if var_name == "MORGAN1024":

                def fingerprint_morgan1024(mol, atom):
                    return SimilarityMaps.GetMorganFingerprint(mol, atom, 4, 'bv', 1024)

                return fingerprint_morgan1024

            elif var_name == "MORGAN2048_1":

                def fingerprint_morgan2048_1(mol, atom):
                    return SimilarityMaps.GetMorganFingerprint(mol, atom, 1, 'bv', 2048)


                return fingerprint_morgan2048_1

            elif var_name == "MORGAN2048_2":

                def fingerprint_morgan2048_2(mol, atom):
                    return SimilarityMaps.GetMorganFingerprint(mol, atom, 2, 'bv', 2048)

                return fingerprint_morgan2048_2

            elif var_name == "MORGAN2048_3":

                def fingerprint_morgan2048_3(mol, atom):
                    return SimilarityMaps.GetMorganFingerprint(mol, atom, 3, 'bv', 2048)


                return fingerprint_morgan2048_3

            elif var_name == "MORGAN2048_4":

                def fingerprint_morgan2048_4(mol, atom):
                    return SimilarityMaps.GetMorganFingerprint(mol, atom, 4, 'bv', 2048)

                return fingerprint_morgan2048_4

            elif var_name == "MORGAN2048_5":

                def fingerprint_morgan2048_5(mol, atom):
                    return SimilarityMaps.GetMorganFingerprint(mol, atom, 5, 'bv', 2048)

                return fingerprint_morgan2048_5

            elif var_name == "MORGAN2048_6":

                def fingerprint_morgan2048_6(mol, atom):
                    return SimilarityMaps.GetMorganFingerprint(mol, atom, 6, 'bv', 2048)

                return fingerprint_morgan2048_6

            else: return None

    # Generate the png similarity diagrams for the test compounds.
    def maps(self, directory):

        if self.fingerprinter is None or self.probability_func is None: return  # silent return.

        diagram_total = len(self.data.get_field("ID"))
        self.log.info("Generating %d Similarity Diagrams in %s.......", diagram_total, directory)

        diagram_count = 0
        for idx in range(len(self.data.get_field("SMILE"))):
            mol = Chem.MolFromSmiles(self.data.get_field("SMILE")[idx])
            fig, weight = SimilarityMaps.GetSimilarityMapForModel(mol,
                                                                  self.fingerprinter,
                                                                  self.probability_func,
                                                                  colorMap=cm.bwr)
            graph_file_name = self.data.get_field("ID")[idx] + "_sim_" + self.model.model_postfix() + ".png"
            graph_path_name = os.path.join(directory, graph_file_name)
            fig.savefig(graph_path_name, bbox_inches="tight")
            plt.close(fig)  # release memory
            diagram_count += 1
            progress_line = "Processing similarity diagram {}/{}\r".format(diagram_count, diagram_total)
            sys.stdout.write(progress_line)
            sys.stdout.flush()





class OSMDragonMap(object):

    def __init__(self, model, data, func):

        self.args = model.args
        self.log = model.log
        self.model = model
        self.data = data

        self.morgan_func = self.check_args()
        self.probability_func = func

    def check_args(self):

        model_args = self.model.model_arguments()
        if len(model_args["INDEPENDENT"]) == 1:             # Only one fingerprint

            var_name = model_args["INDEPENDENT"][0]["VARIABLE"]

            if var_name == "DRAGON":

                return None

            else: return None

    # Generate the png morgan diagram for test compounds.
    def maps(self, directory):

        if self.morgan_func is None or self.probability_func is None: return  # silent return.


    def calc_derivative(self, data, func):

        step_size = 0.01
        vector_size = data.shape[1]
        # Maybe use column median here.
        central_value = np.identity(vector_size) * 0.5
        derivatives = np.zeros(vector_size)

        for idx in range(vector_size):
            central_value[idx] += step_size
            up_prob = func(central_value)
            central_value[idx] -= 2 * step_size
            dn_prob = func(central_value)
            central_value[idx] += step_size
            derivatives[idx] = (up_prob - dn_prob)

        return derivatives


    def calc_sensitivity(self, data, func):

        sensitivity = data * self.calc_derivative(data, func)
        return sensitivity



