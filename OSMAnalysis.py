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
        return np.asarray(m).reshape(-1)  # as array

    def calc_step_min(self, data):
        return np.nanmin(data, axis=0)   # as matrix

    def calc_max(self, data):
        m = np.nanmax(data, axis=0)
        return np.asarray(m).reshape(-1)

    def calc_step_max(self, data):
        return np.nanmax(data, axis=0)

    def calc_median(self, data):
        m = np.median(data, axis=0)
        return np.asarray(m).reshape(-1)

    def calc_step_median(self, data):
        return np.median(data, axis=0)

    def calc_mean(self, data):
        m = np.mean(data, axis=0)
        return np.asarray(m).reshape(-1)

    def calc_step_mean(self, data):
        return np.mean(data, axis=0)

    def calc_range(self, min, max):
        return max - min

    def calc_abs_sensitivity(self, func, data, steps):

        median_data = self.calc_step_median(data)
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

                    abs_prob = np.abs(sens_prob - median_prob)

                    step_prob += abs_prob

                sensitivity[idx] = step_prob

        sens_max = sensitivity.max()

        return sensitivity / sens_max


    def calc_step_sensitivity(self, func, data, steps):

        reference_data = self.calc_step_min(data)
        max_data = self.calc_max(data)
        min_data = self.calc_min(data)
        median_data = self.calc_median(data)
        mean_data = self.calc_mean(data)
        range_data = self.calc_range(self.calc_min(data), self.calc_max(data))

        ref_prob_matrix = func(reference_data)
        ref_prob = ref_prob_matrix[0][0]

        sensitivity = np.zeros((max_data.shape[0], steps+7))

        for idx in range(len(range_data)):

            if range_data[idx] != 0:

                step_size = range_data[idx] / steps
                sens_data = np.asarray(reference_data.copy()).reshape(-1)

                sensitivity[idx][0] = min_data[idx]
                sensitivity[idx][1] = max_data[idx]
                sensitivity[idx][2] = step_size
                sensitivity[idx][3] = median_data[idx]
                sensitivity[idx][4] = mean_data[idx]

                sum_abs = 0.0

                for step_idx in range(int(steps + 1)):

                    sens_data[idx] = min_data[idx] + (step_size * step_idx)

                    sens_matrix = sens_data.reshape(reference_data.shape)

                    sens_prob_matrix = func(sens_matrix)

                    sens_prob = sens_prob_matrix[0][0]

                    step_prob = sens_prob - ref_prob

                    sum_abs += np.abs(step_prob)

                    sensitivity[idx][step_idx + 6] = step_prob

                sensitivity[idx][5] = sum_abs

        return sensitivity

    def calc_sens_derivative(self, func, data, stepsize):

        reference_data = self.calc_step_mean(data)
        max_data = self.calc_max(data)
        min_data = self.calc_min(data)
        range_data = self.calc_range(min_data, max_data)

        ref_prob_matrix = func(reference_data)
        ref_prob = ref_prob_matrix[0][0]

        sensitivity = np.zeros(max_data.shape)

        for idx in range(len(range_data)):

            if range_data[idx] != 0:

                sens_data = np.asarray(reference_data.copy()).reshape(-1)

                sens_data[idx] += stepsize

                sens_matrix = sens_data.reshape(reference_data.shape)

                sens_prob_matrix = func(sens_matrix)

                sens_prob = sens_prob_matrix[0][0]

                step_prob = sens_prob - ref_prob

                sensitivity[idx] = step_prob

        sens_max = sensitivity.max()

        return sensitivity / sens_max

    def calc_partial_derivative(self, func, data, stepsize):

        reference_data = self.calc_step_mean(data)
        max_data = self.calc_max(data)
        min_data = self.calc_min(data)
        range_data = self.calc_range(min_data, max_data)

        ref_prob_matrix = func(reference_data)
        ref_prob = ref_prob_matrix[0][0]

        sensitivity = np.zeros((range_data.shape[0],range_data.shape[0]))

        sens_data = np.asarray(reference_data.copy()).reshape(-1)

        for row in range(len(range_data)):

            if range_data[row] != 0:

                sens_data[row] += stepsize

                for col in range(len(range_data)):

                    if range_data[col] != 0:

                        sens_data[col] += stepsize

                        sens_matrix = sens_data.reshape(reference_data.shape)

                        sens_prob_matrix = func(sens_matrix)

                        sens_data[col] -= stepsize

                        sens_prob = sens_prob_matrix[0][0]

                        step_prob = sens_prob - ref_prob

                        sensitivity[row][col] = step_prob

                sens_data[row] -= stepsize


            progress_line = "Processing partial derivative row {}/{}\r".format(row, len(range_data))
            sys.stdout.write(progress_line)
            sys.stdout.flush()

        sens_max = sensitivity.max()

        return sensitivity / sens_max

    def calc_derivative(self, func, data, step_size):

        max_data = self.calc_max(data)
        sensitivity = np.zeros(max_data.shape)
        record_count = np.zeros(max_data.shape)

        for record in data:

            record_prob_matrix = func(record)
            record_prob = record_prob_matrix[0][0]

            sens_data = np.asarray(record.copy()).reshape(-1)

            for idx in range(len(sens_data)):

                if sens_data[idx] != 0:
                
                    sens_data[idx] += step_size

                    sens_matrix = sens_data.reshape(record.shape)
                    sens_prob_matrix = func(sens_matrix)
                    sens_prob = sens_prob_matrix[0][0]

                    sens_data[idx] -= step_size

                    prob_diff = sens_prob-record_prob

                    sensitivity[idx] += prob_diff
                    record_count[idx] += 1
                    
        sens_max = sensitivity.max()

        return sensitivity / sens_max

class OSMDragonSensitivity(OSMSensitivity):
    def __init__(self, args, log):
        super(OSMDragonSensitivity, self).__init__(args, log)

    def calc_dragon_sensitivity(self, func, data, steps, dragon_field_list, description_list):

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
            sens_list.append([field_list[idx], description_list[idx], idx, sens_array[idx]])

        sorted_list = sorted(sens_list, key=lambda x: -x[3])

        return sorted_list

    def calc_dragon_step_sensitivity(self, func, data, steps, dragon_field_list, description_list):

        sens_array = self.calc_step_sensitivity(func, data, steps)

        # remove the SMILE field

        field_list = dragon_field_list
        field_list.pop(0)

        if sens_array.shape[0] != len(field_list):
            self.log.error("Mismatch, dragon_field_list size: %d, sensitivity vector size: %d"
                           , len(field_list), sens_array.shape[0])
            sys.exit()

        sens_list = []
        for idx in range(sens_array.shape[0]):
            row_abs_sum = sens_array[idx][5]
            sens_list.append([field_list[idx], description_list[idx], idx, row_abs_sum, sens_array[idx]])

        sorted_list = sorted(sens_list, key=lambda x: -x[3])

        return sorted_list


    def calc_dragon_derivative(self, func, data, stepsize, dragon_field_list, description_list):

        sens_array = self.calc_sens_derivative(func, data, stepsize)

        # remove the SMILE field

        field_list = dragon_field_list
        field_list.pop(0)

        if len(sens_array) != len(field_list):
            self.log.error("Mismatch, dragon_field_list size: %d, sensitivity vector size: %d"
                           , len(field_list), len(sens_array))
            sys.exit()

        sens_list = []
        for idx in range(len(sens_array)):
            sens_list.append([field_list[idx], description_list[idx], idx, sens_array[idx]])

        sorted_list = sorted(sens_list, key=lambda x: -x[3])

        return sorted_list

    def calc_dragon_partial(self, func, data, stepsize, dragon_field_list, description_list):

        partial_matrix = self.calc_partial_derivative(func, data, stepsize)

        # remove the SMILE field

        field_list = dragon_field_list
        field_list.pop(0)

        if partial_matrix.shape[0] != len(field_list):
            self.log.error("Mismatch, dragon_field_list size: %d, partial square matrix size: %d"
                           , len(field_list), partial_matrix.shape[0])
            sys.exit()

        return [field_list, description_list, partial_matrix]

class OSMTruncDragonSensitivity(OSMSensitivity):
    def __init__(self, args, log):
        super(OSMTruncDragonSensitivity, self).__init__(args, log)

    def calc_trunc_dragon_sensitivity(self, func, data, steps, field_list, description_list):

        sens_array = self.calc_abs_sensitivity(func, data, steps)

        if len(sens_array) != len(field_list):
            self.log.error("Mismatch, truncated dragon_field_list size: %d, sensitivity vector size: %d"
                           , len(field_list), len(sens_array))
            sys.exit()

        sens_list = []
        for idx in range(len(sens_array)):
            sens_list.append([field_list[idx], description_list[idx], idx, sens_array[idx]])

        sorted_list = sorted(sens_list, key=lambda x: -x[3])

        return sorted_list

    def calc_trunc_dragon_derivative(self, func, data, step_size, field_list, description_list):

        sens_array = self.calc_sens_derivative(func, data, step_size)

        if len(sens_array) != len(field_list):
            self.log.error("Mismatch, truncated dragon_field_list size: %d, sensitivity vector size: %d"
                           , len(field_list), len(sens_array))
            sys.exit()

        sens_list = []
        for idx in range(len(sens_array)):
            sens_list.append([field_list[idx], description_list[idx], idx, sens_array[idx]])

        sorted_list = sorted(sens_list, key=lambda x: -x[3])

        return sorted_list

    def calc_trunc_dragon_step_sensitivity(self, func, data, steps, field_list, description_list):

        sens_array = self.calc_step_sensitivity(func, data, steps)

        if sens_array.shape[0] != len(field_list):
            self.log.error("Mismatch, truncated dragon_field_list size: %d, sensitivity vector size: %d"
                           , len(field_list), sens_array.shape[0])
            sys.exit()

        sens_list = []
        for idx in range(sens_array.shape[0]):
            row_abs_sum = sens_array[idx][5]
            sens_list.append([field_list[idx], description_list[idx], idx, row_abs_sum, sens_array[idx]])

        sorted_list = sorted(sens_list, key=lambda x: -x[3])

        return sorted_list

    def calc_trunc_dragon_partial(self, func, data, step_size, field_list, description_list):

        partial_matrix = self.calc_partial_derivative(func, data, step_size)

        if partial_matrix.shape[0] != len(field_list):
            self.log.error("Mismatch, truncated dragon_field_list size: %d, partial square matrix size: %d"
                           , len(field_list), partial_matrix.shape[0])
            sys.exit()

        return [field_list, description_list, partial_matrix]
