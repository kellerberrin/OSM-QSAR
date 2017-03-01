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

from OSMBase import OSMBaseModel

import csv
import time

import math
import scipy.stats as st
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize

# ============================================================================
# The Classification Results Presentation Object.
# ============================================================================


class OSMClassification(OSMBaseModel):


    def __init__(self, args, log):
        super(OSMClassification, self).__init__(args, log)


        # Shallow copies of the runtime environment.
        self.log = log
        self.args = args

#####################################################################################
#
# Virtual member functions called from OSMBase
#
#####################################################################################


    def model_log_statistics(self, model, train, test):
#        self.log_train_statistics(model, train)
        self.log_test_statistics(model, test)

    def model_accuracy(self, predictions):
        predict = predictions["prediction"]
        actual = predictions["actual"]

        print("actual", actual, "prediction", predictions)

        auc_stat = auc(predict, actual)

        # Return the model analysis statistics in a dictionary.

        return {"AUC": auc_stat, "actual": actual, "predict" : predict}

    def model_write_statistics(self, model, train, test):

        # Open the statistics file and append the model results statistics.

        try:

            with open(self.args.statsFilename, 'a') as stats_file:

                line = "****************,Classification,******************\n"
                stats_file.write(line)
                line = "Model, {}\n".format(self.model_name())
                stats_file.write(line)
                line = "Runtime, {}\n".format(time.asctime(time.localtime(time.time())))
                stats_file.write(line)
                line = "CPUtime, {}\n".format(time.clock())
                stats_file.write(line)
                line = "++++++++++++++++,Test Statistics,++++++++++++++++\n"
                stats_file.write(line)
                line = "AUC, {}\n".format(self.test_stats["AUC"])
                stats_file.write(line)
                line = "ID, Actual_Class, Pred._Class, SMILE\n"
                stats_file.write(line)

                for idx in range(len(test["ID"])):
                    line = "{}, {}, {}, {}\n".format(test["ID"][idx],
                                                     self.test_stats["actual"][idx],
                                                     self.test_stats["predict"][idx],
                                                     test["SMILE"][idx])
                    stats_file.write(line)



        except IOError:
            self.log.error("Problem writing to statistics file %s, check path and permissions",
                           self.args.statsFilename)


    def model_binary_labels(self, data):
        return label_binarize(self.model_classify_pEC50(data), classes=self.model_enumerate_classes())


#####################################################################################
#
# Local member functions
#
#####################################################################################


    def log_train_statistics(self, model, train):

        self.log.info("Training Compounds Area Under Curve (AUC): %f", self.train_stats["AUC"])

    # Display the classification results and write to the log file.
    def log_test_statistics(self, model, data):
        """Display all the calculated statistics for each model; run"""

        self.log.info("Test Compounds Area Under Curve: %f", self.test_stats["AUC"])
        self.log.info("ID, Actual Class, Pred. Class")
        self.log.info("===================================================================================")

        for idx in range(len(data["ID"])):
            self.log.info("%s, %s, %s", data["ID"][idx],
                          self.test_stats["actual"][idx],
                          self.test_stats["predict"][idx])

