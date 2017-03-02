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


import csv
import time

import math
import scipy.stats as st
from sklearn.metrics import auc

from OSMBase import OSMBaseModel
from OSMUtility import OSMUtility

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

    def model_classification_results(self, model, train, test):
        self.train_predictions = self.model_prediction(model,train)  # Returns a dict. with "prediction" and "actual"
        self.train_probability = self.model_probability(model,train)  # Returns a dict. with "probability"
        self.train_stats = self.model_accuracy(self.train_predictions, self.train_probability)  # dictionary of stats

        self.test_predictions = self.model_prediction(model, test)  # Returns a dict. with "prediction" and "actual"
        self.test_probability = self.model_probability(model,test)  # Returns a dict. with "probability"
        self.test_stats = self.model_accuracy(self.test_predictions, self.test_probability)  # dictionary of stats
        # Send statistics to the console and log file.
        self.model_log_statistics(model, train, test)
        # Generate graphics (only if the virtual function defined at model level).
        self.model_graphics(model, train, test)
        # Append statistics to the stats file.
        self.model_write_statistics(model, train, test)


    def model_log_statistics(self, model, train, test):
        self.log_train_statistics(model, train)
        self.log_test_statistics(model, test)

    def model_accuracy(self, predictions, probability):
        predict = predictions["prediction"]
        actual = predictions["actual"]
        probabilities = probability["probability"]
        inv_probability = [1-x[0] for x in probabilities]  # only interested in the first column ("active")
        # Sort rankings.
        probability_ranks = st.rankdata(inv_probability, method='average')

#        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
#        auc_stat = auc(predict, actual)
        auc_stat = 0

        actual_text = OSMUtility.one_hot_text(actual, self.args.activeNmols)
        predict_text = OSMUtility.probability_text(probabilities, self.args.activeNmols)

        # Return the model analysis statistics in a dictionary.
        return {"AUC": auc_stat, "actual": actual, "predict" : predict, "prob_rank" : probability_ranks,
                "actual_text" : actual_text, "predict_text" : predict_text }

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
                line = "ID, Actual_Class, Pred_Class "
                classes = OSMUtility.enumerate_classes(self.args.activeNmols)
                for cls in classes:
                    line += ", Prob_" + cls
                line += ", SMILE\n"
                stats_file.write(line)

                for idx in range(len(test["ID"])):
                    line = "{}, {}, {}".format(test["ID"][idx],
                                                     self.test_stats["actual_text"][idx],
                                                     self.test_stats["predict_text"][idx])

                    for cls_idx in range(self.test_probability["probability"][idx].size):
                        line += ", {}".format(self.test_probability["probability"][idx][cls_idx])

                    line += ", {}\n".format(test["SMILE"][idx])

                    stats_file.write(line)



        except IOError:
            self.log.error("Problem writing to statistics file %s, check path and permissions",
                           self.args.statsFilename)




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
        self.log.info("ID, Actual Class, Pred. Class, Prob. Active, Prob. Rank")
        self.log.info("===================================================================================")

        for idx in range(len(data["ID"])):
            self.log.info("%s, %s, %s, %f, %d", data["ID"][idx],
                          self.test_stats["actual_text"][idx],
                          self.test_stats["predict_text"][idx],
                          self.test_probability["probability"][idx][0],
                          self.test_stats["prob_rank"][idx])

