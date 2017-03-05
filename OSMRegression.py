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

import os
import time
import math

import scipy.stats as st

from OSMBase import OSMBaseModel
from OSMUtility import OSMUtility


# ============================================================================
# The Regression Results Presentation Object.
# ============================================================================


class OSMRegression(OSMBaseModel):
    def __init__(self, args, log):
        super(OSMRegression, self).__init__(args, log)

        # Shallow copies of the runtime environment.
        self.log = log
        self.args = args

    #####################################################################################
    #
    # Virtual member functions called from OSMBase
    #
    #####################################################################################

    def model_is_regression(self):
        return True

    def model_classification_results(self):
        self.train_predictions = self.model_prediction(self.data.training())  # Returns a dict. with "prediction" and "actual"
        self.train_stats = self.model_accuracy(self.train_predictions)  # Returns a dictionary of accuracy tests.

        self.test_predictions = self.model_prediction(self.data.testing())
        self.test_stats = self.model_accuracy(self.test_predictions)
        # Send statistics to the console and log file.
        self.model_log_statistics()
        # Generate graphics (only if the virtual function defined at model level).
        self.model_graphics()
        # Append statistics to the stats files.
        self.model_write_statistics()

    def model_log_statistics(self):
        self.log_train_statistics(self.data.training())
        self.log_test_statistics(self.data.testing())

    def model_write_statistics(self):
        self.write_statistics(self.data.training(), self.train_stats, self.train_predictions, self.args.trainDirectory)
        self.write_statistics(self.data.testing(), self.test_stats, self.test_predictions, self.args.testDirectory)

    def model_accuracy(self, predictions):
        predict = predictions["prediction"]
        actual = predictions["actual"]

        MUE = 0
        RMSE = 0
        for i in range(len(predict)):
            diff = abs(predict[i] - actual[i])
            MUE += diff
            RMSE = diff * diff

        MUE = MUE / len(predict)
        RMSE = RMSE / len(predict)
        RMSE = math.sqrt(RMSE)

        # Sort rankings.

        predict_ranks = st.rankdata(predict, method='average')
        actual_ranks = st.rankdata(actual, method='average')

        # generate Kendel rank correlation coefficient

        kendall = {}
        tau, p_value = st.kendalltau(actual_ranks, predict_ranks)
        kendall["tau"] = tau
        kendall["p-value"] = p_value
        spearman = {}
        rho, p_value = st.spearmanr(actual_ranks, predict_ranks)
        spearman["rho"] = rho
        spearman["p-value"] = p_value

        # Return the model analysis statistics in a dictionary.
        return {"MUE": MUE, "RMSE": RMSE, "predict_ranks": predict_ranks,
                "actual_ranks": actual_ranks, "kendall": kendall, "spearman": spearman}

    #####################################################################################
    #
    # Local member functions
    #
    #####################################################################################

    def log_train_statistics(self, train):

        self.log.info("Training Compounds pIC50 Mean Unsigned Error (MUE): %f", self.train_stats["MUE"])
        self.log.info("Training Compounds pIC50 RMS Error: %f", self.train_stats["RMSE"])

    # Display the classification results and write to the log file.
    def log_test_statistics(self, test):
        """Display all the calculated statistics for each model; run"""

        self.log.info("Test Compounds pIC50 Mean Unsigned Error (MUE): %f", self.test_stats["MUE"])
        self.log.info("Test Compounds pIC50 RMS Error: %f", self.test_stats["RMSE"])

        self.log.info("Test Compounds Kendall's Rank Coefficient (tau): %f, p-value: %f",
                      self.test_stats["kendall"]["tau"], self.test_stats["kendall"]["p-value"])
        self.log.info("Test Compounds Spearman Coefficient (rho): %f, p-value: %f",
                      self.test_stats["spearman"]["rho"], self.test_stats["spearman"]["p-value"])
        self.log.info("Test Compounds pIC50 Mean Unsigned Error (MUE): %f", self.test_stats["MUE"])
        self.log.info("Test Compounds Results")
        self.log.info("ID, Tested Rank, Pred. Rank, Tested pIC50, Pred. pIC50")
        self.log.info("======================================================")

        for idx in range(len(test.get_field("ID"))):
            self.log.info("%s, %d, %d, %f, %f", test.get_field("ID")[idx],
                          self.test_stats["actual_ranks"][idx],
                          self.test_stats["predict_ranks"][idx],
                          self.test_predictions["actual"][idx],
                          self.test_predictions["prediction"][idx])

    def write_statistics(self, data, statistics, predictions, directory):

        # Open the statistics file and append the model results statistics.

        stats_filename = os.path.join(directory, self.args.statsFilename)
        try:

            with open(stats_filename, 'a') as stats_file:

                line = "****************,Classification,******************\n"
                stats_file.write(line)
                line = "Model, {}\n".format(self.model_name())
                stats_file.write(line)
                line = "Runtime, {}\n".format(time.asctime(time.localtime(time.time())))
                stats_file.write(line)
                line = "CPUtime, {}\n".format(time.clock())
                stats_file.write(line)
                line = "++++++++++++++++,Statistics,++++++++++++++++\n"
                stats_file.write(line)
                line = "MUE, {}\n".format(statistics["MUE"])
                stats_file.write(line)
                line = "RMSE, {}\n".format(statistics["RMSE"])
                stats_file.write(line)
                line = "Kendall, {}, {}\n".format(statistics["kendall"]["tau"],
                                                  statistics["kendall"]["p-value"])
                stats_file.write(line)
                line = "Spearman, {}, {}\n".format(statistics["spearman"]["rho"],
                                                   statistics["spearman"]["p-value"])
                stats_file.write(line)
                line = "ID, Rank, Pred_Rank, Tested_pEC50, Pred_pEC50, Tested_Active, Pred_Active, SMILE\n"
                stats_file.write(line)

                for idx in range(len(data.get_field("ID"))):
                    line = "{}, {}, {}, {}, {}, {}\n".format(data.get_field("ID")[idx],
                                                             statistics["actual_ranks"][idx],
                                                             statistics["predict_ranks"][idx],
                                                             predictions["actual"][idx],
                                                             predictions["prediction"][idx],
                                                             data.get_field("SMILE")[idx])
                    stats_file.write(line)

        except IOError:
            self.log.error("Problem writing to statistics file %s, check path and permissions", stats_filename)
