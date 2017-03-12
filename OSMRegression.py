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
        self.log_train_statistics(self.data.training(), self.train_stats, self.train_predictions)
        self.log_test_statistics(self.data.testing(), self.test_stats, self.test_predictions)

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


    def model_prediction_records(self, data, predictions, statistics):

        prediction_list = []
        for idx in range(len(data.get_field("ID"))):
            prediction_record = []
            prediction_record.append(data.get_field("ID")[idx])
            prediction_record.append(statistics["actual_ranks"][idx])
            prediction_record.append(statistics["predict_ranks"][idx])
            prediction_record.append(predictions["actual"][idx])
            prediction_record.append(predictions["prediction"][idx])
            prediction_record.append(data.get_field("SMILE")[idx])

            prediction_list.append(prediction_record)

        # Sort by actual ranking.
        sorted_predict_list= sorted(prediction_list, key=lambda predict_record: predict_record[1])

        return sorted_predict_list


    #####################################################################################
    #
    # Local member functions
    #
    #####################################################################################

    def log_train_statistics(self, train, statistics, predictions):

        self.log.info("Training Compounds Mean Unsigned Error (MUE): %f", self.train_stats["MUE"])
        self.log.info("Training Compounds RMS Error: %f", self.train_stats["RMSE"])

    # Display the classification results and write to the log file.
    def log_test_statistics(self, data, statistics, predictions):
        """Display all the calculated statistics for each model; run"""

        independent_list = []
        for var in self.model_arguments()["INDEPENDENT"]:
            independent_list.append(var["VARIABLE"])
        dependent_var = self.model_arguments()["DEPENDENT"]["VARIABLE"]

        self.log.info("Dependent (Target) Variable: %s", dependent_var)
        for var in independent_list:
            self.log.info("Independent (Input) Variable(s): %s", var)

        self.log.info("Training Epochs: %d", self.model_epochs())

        self.log.info("Test Compounds %s Mean Unsigned Error (MUE): %f", dependent_var, statistics["MUE"])
        self.log.info("Test Compounds %s RMS Error: %f", dependent_var, statistics["RMSE"])

        self.log.info("Test Compounds Kendall's Rank Coefficient (tau): %f, p-value: %f",
                      statistics["kendall"]["tau"], statistics["kendall"]["p-value"])
        self.log.info("Test Compounds Spearman Coefficient (rho): %f, p-value: %f",
                      statistics["spearman"]["rho"], statistics["spearman"]["p-value"])
        self.log.info("Test Compounds %s Mean Unsigned Error (MUE): %f", dependent_var, statistics["MUE"])
        self.log.info("Test Compounds Results")
        self.log.info("ID, Tested Rank, Pred. Rank, Tested %s, Pred. %s", dependent_var, dependent_var)
        self.log.info("======================================================")

        predict_list = self.model_prediction_records(data, predictions, statistics)

        for idx in range(len(predict_list)):
            line = "{:10s} {:4.1f} {:4.1f} {:10.4f} {:10.4f}".format(predict_list[idx][0],
                                                                 predict_list[idx][1],
                                                                 predict_list[idx][2],
                                                                 predict_list[idx][3],
                                                                 predict_list[idx][4])
            self.log.info(line)

    def write_statistics(self, data, statistics, predictions, directory):

        # Open the statistics file and append the model results statistics.

        stats_filename = os.path.join(directory, self.args.statsFilename)
        independent_list = []
        for var in self.model_arguments()["INDEPENDENT"]:
            independent_list.append(var["VARIABLE"])
        dependent_var = self.model_arguments()["DEPENDENT"]["VARIABLE"]
        try:

            with open(stats_filename, 'a') as stats_file:

                line = "****************,Classification,******************\n"
                stats_file.write(line)
                line = "Model, {}\n".format(self.model_name())
                stats_file.write(line)
                line = "DependentVar(Target), {}\n".format(dependent_var)
                stats_file.write(line)
                for var in independent_list:
                    line = "IndependentVar(Input), {}\n".format(var)
                    stats_file.write(line)
                line = "TrainingEpochs, {}\n".format(self.model_epochs())
                stats_file.write(line)
                line = "Runtime, {}\n".format(time.asctime(time.localtime(time.time())))
                stats_file.write(line)
                line = "CPUtime, {}\n".format(time.clock())
                stats_file.write(line)
                line = "++++++++++++++++,Test_Statistics,++++++++++++++++\n"
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
                line = "++++++++++++++++,Compound_Statistics,++++++++++++++++\n"
                stats_file.write(line)
                line = "ID, Rank, Pred_Rank, Tested_{}, Pred_{}, SMILE\n".format(dependent_var, dependent_var)
                stats_file.write(line)

                predict_list = self.model_prediction_records(data, predictions, statistics)

                for idx in range(len(data.get_field("ID"))):
                    line = "{}, {}, {}, {}, {}, {}\n".format(predict_list[idx][0],
                                                             predict_list[idx][1],
                                                             predict_list[idx][2],
                                                             predict_list[idx][3],
                                                             predict_list[idx][4],
                                                             predict_list[idx][5])
                    stats_file.write(line)

        except IOError:
            self.log.error("Problem writing to statistics file %s, check path and permissions", stats_filename)
