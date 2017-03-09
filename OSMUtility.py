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

import numpy

from sklearn.preprocessing import label_binarize

# ============================================================================
# This source file implements data manipulation and other utility objects.
# These objects are not part of the OSMBase object hierarchy.
# ============================================================================

class OSMUtility(object):

    def __init__(self): pass


    # Return the one hot classifications including one hot singletons.
    @staticmethod
    def data_one_hot(data, training_classes):
        hot_one_labels = label_binarize(data, training_classes)
        return hot_one_labels

    # Flatten a list of one hot singletons to a binary list, otherwise SKLearn classifiers complain.
    @staticmethod
    def flatten_one_hot(one_hot_labels):

        if len(one_hot_labels) == 0:
            return one_hot_labels

        if isinstance(one_hot_labels[0], list) and len(one_hot_labels[0]) == 1:
            one_hot_labels = one_hot_labels.flatten()

        return one_hot_labels

    # Present data to a SKLearn classifier with flattened binary singletons.
    @staticmethod
    def data_classify(data, training_classes):
        return OSMUtility.flatten_one_hot(OSMUtility.data_one_hot(data, training_classes))