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


class OSMSensitivity(object):

    def __init__(self, args, log):
        self.args = args
        self.log = log

    # The data is assumed to be a numpy matrix.
    # Firstly we calculate the maximum and minimum of each column.

    def calc_min(self, data):
        m = np.nanmin(data, axis=0)
        return np.asarray(m).reshape(-1)

    def calc_max(self, data):
        m = np.nanmax(data, axis=0)
        return np.asarray(m).reshape(-1)

    def calc_median(self, data):
        return np.median(data, axis=0)

    def calc_range(self, min, max):
        return max - min

    def calc_abs_sensitivity(self, func, data, steps):

        median_data = self.calc_median(data)
        max_data = self.calc_max(data)
        min_data = self.calc_min(data)
        range_data = self.calc_range(min_data, max_data)

        median_prob_matrix = func(median_data)
        median_prob = median_prob_matrix[0][0]

        sensitivity = np.zeros(max_data.shape)

        for idx in range(len(range_data)):

            if range_data[idx] != 0:

                step_size = range_data[idx] / steps
                sens_data = np.asarray(median_data.copy()).reshape(-1)
                step_prob = 0.0

                for step_idx in range(int(steps)+1):

                    sens_data[idx] = min_data[idx] + (step_size * step_idx)

                    sens_matrix = sens_data.reshape(median_data.shape)

                    sens_prob_matrix = func(sens_matrix)

                    sens_prob = sens_prob_matrix[0][0]

                    abs_prob = np.sum(np.absolute(median_prob - sens_prob))

                    step_prob += abs_prob

                sensitivity[idx] = step_prob

        sens_max = sensitivity.max()

        return sensitivity / sens_max

    def calc_partial_derivative(self, func, data, step_size):

        max_data = self.calc_max(data)
        sensitivity = np.zeros(max_data.shape)

        for record in data:

            record_prob_matrix = func(record)
            record_prob = record_prob_matrix[0][0]

            sens_data = np.asarray(record.copy()).reshape(-1)

            for idx in range(len(sens_data)):

                sens_data[idx] += step_size

                sens_matrix = sens_data.reshape(record.shape)
                sens_prob_matrix = func(sens_matrix)
                sens_prob = sens_prob_matrix[0][0]

                sens_data[idx] -= step_size

                prob_diff = sens_prob-record_prob

                sensitivity[idx] += prob_diff

        sens_max = sensitivity.max()

        return sensitivity / sens_max

class OSMDragonSensitivity(OSMSensitivity):
    def __init__(self, args, log):
        super(OSMDragonSensitivity, self).__init__(args, log)

    def calc_dragon_sensitivity(self, func, data, steps, dragon_field_list):

        sens_array = self.calc_abs_sensitivity(func, data, steps)

        # remove the SMILE field

        field_list = dragon_field_list
        field_list.pop(0)

        if len(sens_array) != len(field_list):
            self.log.error("Mismatch, dragon_field_list size: %d, sensitivity vector size: %d"
                           , len(field_list), len(sens_array))
            sys.exit()

        sens_list = []
        for idx in range(len(sens_array)):
            sens_list.append([field_list[idx], idx, sens_array[idx]])

        sorted_list = sorted(sens_list, key=lambda x: -x[2])

        return sorted_list

    def calc_dragon_derivative(self, func, data, steps, dragon_field_list):

        sens_array = self.calc_partial_derivative(func, data, steps)

        # remove the SMILE field

        field_list = dragon_field_list
        field_list.pop(0)

        if len(sens_array) != len(field_list):
            self.log.error("Mismatch, dragon_field_list size: %d, sensitivity vector size: %d"
                           , len(field_list), len(sens_array))
            sys.exit()

        sens_list = []
        for idx in range(len(sens_array)):
            sens_list.append([field_list[idx], idx, sens_array[idx]])

        sorted_list = sorted(sens_list, key=lambda x: -x[2])

        return sorted_list

class OSMTruncDragonSensitivity(OSMSensitivity):
    def __init__(self, args, log):
        super(OSMTruncDragonSensitivity, self).__init__(args, log)

    def calc_trunc_dragon_sensitivity(self, func, data, steps, field_list):

        sens_array = self.calc_abs_sensitivity(func, data, steps)

        if len(sens_array) != len(field_list):
            self.log.error("Mismatch, truncated dragon_field_list size: %d, sensitivity vector size: %d"
                           , len(field_list), len(sens_array))
            sys.exit()

        sens_list = []
        for idx in range(len(sens_array)):
            sens_list.append([field_list[idx], idx, sens_array[idx]])

        sorted_list = sorted(sens_list, key=lambda x: -x[2])

        return sorted_list

    def calc_trunc_dragon_derivative(self, func, data, steps, field_list):

        sens_array = self.calc_partial_derivative(func, data, steps)

        if len(sens_array) != len(field_list):
            self.log.error("Mismatch, truncated dragon_field_list size: %d, sensitivity vector size: %d"
                           , len(field_list), len(sens_array))
            sys.exit()

        sens_list = []
        for idx in range(len(sens_array)):
            sens_list.append([field_list[idx], idx, sens_array[idx]])

        sorted_list = sorted(sens_list, key=lambda x: -x[2])

        return sorted_list
