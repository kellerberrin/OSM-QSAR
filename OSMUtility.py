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

    # Accepts a list (array) of pEC50 values and returns a list (array) of classification text labels.
    #  args.activeNmols is a list (array) of pEC50 nMol potency sorted tuples [(potency, "class"), ...]
    @staticmethod
    def data_text(value_array, classes):

        class_array = []
        for x in value_array:

            if x <= classes[0][0]:
                class_str = classes[0][1]
            else:
                class_str = "inactive"

            if len(classes) > 1:
                for idx in range(len(classes) - 1):
                    if x > classes[idx][0] and x <= classes[idx + 1][0]:
                        class_str = classes[idx + 1][1]

            class_array.append(class_str)

        return class_array

    # Returns a text list of defined potency classification classes, including the implied "inactive" class.
    @staticmethod
    def enumerate_classes(potency_classes):

        class_array = []

        for potency_class in potency_classes:
            class_array.append(potency_class[1])

        class_array.append("inactive")

        return class_array

    # Accepts a numpy.ndarray of probabilities shape = (n_samples, n_classes)
    # Returns a text list of potency classes.
    @staticmethod
    def probability_text(prob_list, potency_classes):
        if len(prob_list) == 0:
            return prob_list
        text_classes = OSMUtility.enumerate_classes(potency_classes)
        if isinstance(prob_list, numpy.ndarray):
            class_list = []
            for x in prob_list:
                idx = numpy.argmax(x)
                class_list.append(text_classes[idx])
            return class_list
        else: # assume a list.
            class_list = []
            for x in prob_list:
                idx = x.index(max(x))
                class_list.append(text_classes[idx])
            return class_list

    # Accepts a list (not a numpy.ndarray!) of one hot classifications, or if only 2 potency classes,
    # a single binary list. Returns a text list of potency classes.
    @staticmethod
    def one_hot_text(one_hot_labels, potency_classes):
        if len(one_hot_labels) == 0:
            return one_hot_labels

        text_classes = OSMUtility.enumerate_classes(potency_classes)
        if not isinstance(one_hot_labels[0], list) and len(text_classes) == 2:
            return [text_classes[0 if x == 0 else 1] for x in one_hot_labels]

        elif isinstance(one_hot_labels[0], list) and len(one_hot_labels[0]) == 1 and len(text_classes) == 2:
            return [text_classes[0 if x[0] == 0 else 1] for x in one_hot_labels]

        else:
            class_list = []
            for x in one_hot_labels:
                found_class = False
                for idx in range(len(x)):
                    if x[idx] > 0:
                        class_list.append(text_classes[idx])
                        found_class = True
                        continue
                if not found_class:
                    class_list.append("no classify")

            return class_list

    # Return the one hot classifications including one hot singletons.
    @staticmethod
    def data_one_hot(data, potency_classes):
        hot_one_labels = label_binarize(OSMUtility.data_text(data,potency_classes)
                                        , classes=OSMUtility.enumerate_classes(potency_classes))
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
    def data_classify(data, potency_classes):
        return OSMUtility.flatten_one_hot(OSMUtility.data_one_hot(data, potency_classes))