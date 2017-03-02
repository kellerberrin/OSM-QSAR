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

    def model_classification_results(self, model, train, test):
        self.train_predictions = self.model_prediction(model,train)  # Returns a dict. with "prediction" and "actual"
        self.train_stats = self.model_accuracy(self.train_predictions)  # Returns a dictionary of accuracy tests.

        self.test_predictions = self.model_prediction(model, test)
        self.test_stats = self.model_accuracy(self.test_predictions)
        # Send statistics to the console and log file.
        self.model_log_statistics(model, train, test)
        # Generate graphics (only if the virtual function defined at model level).
        self.model_graphics(model, train, test)
        # Append statistics to the stats file.
        self.model_write_statistics(model, train, test)

    def model_log_statistics(self, model, train, test):
        self.log_train_statistics(model, train)
        self.log_test_statistics(model, test)

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

        #  Active / Inactive classifications

        predict_active = OSMUtility.data_text(predict, self.args.activeNmols)
        actual_active = OSMUtility.data_text(actual, self.args.activeNmols)

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
                "actual_ranks": actual_ranks, "predict_active": predict_active,
                "actual_active": actual_active, "kendall": kendall, "spearman": spearman}

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
                line = "MUE, {}\n".format(self.test_stats["MUE"])
                stats_file.write(line)
                line = "RMSE, {}\n".format(self.test_stats["RMSE"])
                stats_file.write(line)
                line = "Kendall, {}, {}\n".format(self.test_stats["kendall"]["tau"],
                                                  self.test_stats["kendall"]["p-value"])
                stats_file.write(line)
                line = "Spearman, {}, {}\n".format(self.test_stats["spearman"]["rho"],
                                                   self.test_stats["spearman"]["p-value"])
                stats_file.write(line)
                line = "ID, Rank, Pred_Rank, Tested_pEC50, Pred_pEC50, Tested_Active, Pred_Active, SMILE\n"
                stats_file.write(line)

                for idx in range(len(test["ID"])):
                    line = "{}, {}, {}, {}, {}, {}, {}, {}\n".format(test["ID"][idx],
                                                                     self.test_stats["actual_ranks"][idx],
                                                                     self.test_stats["predict_ranks"][idx],
                                                                     self.test_predictions["actual"][idx],
                                                                     self.test_predictions["prediction"][idx],
                                                                     self.test_stats["actual_active"][idx],
                                                                     self.test_stats["predict_active"][idx],
                                                                     test["SMILE"][idx])
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
    
        self.log.info("Training Compounds pEC50 Mean Unsigned Error (MUE): %f", self.train_stats["MUE"])
        self.log.info("Training Compounds pEC50 RMS Error: %f", self.train_stats["RMSE"])


# Display the classification results and write to the log file.
    def log_test_statistics(self, model, data):
        """Display all the calculated statistics for each model; run"""


        self.log.info("Test Compounds pEC50 Mean Unsigned Error (MUE): %f", self.test_stats["MUE"])
        self.log.info("Test Compounds pEC50 RMS Error: %f", self.test_stats["RMSE"])

        self.log.info("Test Compounds Kendall's Rank Coefficient (tau): %f, p-value: %f",
                      self.test_stats["kendall"]["tau"], self.test_stats["kendall"]["p-value"])
        self.log.info("Test Compounds Spearman Coefficient (rho): %f, p-value: %f",
                      self.test_stats["spearman"]["rho"], self.test_stats["spearman"]["p-value"])
        self.log.info("Test Compounds pEC50 Mean Unsigned Error (MUE): %f", self.test_stats["MUE"])
        self.log.info("Test Compounds Results")
        self.log.info("ID, Tested Rank, Pred. Rank, Tested pEC50, Pred. pEC50, Tested Active, Pred. Active")
        self.log.info("===================================================================================")

        for idx in range(len(data["ID"])):
            self.log.info("%s, %d, %d, %f, %f, %s, %s", data["ID"][idx],
                                                        self.test_stats["actual_ranks"][idx],
                                                        self.test_stats["predict_ranks"][idx],
                                                        self.test_predictions["actual"][idx],
                                                        self.test_predictions["prediction"][idx],
                                                        self.test_stats["actual_active"][idx],
                                                        self.test_stats["predict_active"][idx]),




