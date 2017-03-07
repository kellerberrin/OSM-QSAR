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
import sys
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

    def model_is_classifier(self):
        return True

    def model_enumerate_classes(self):

        class_set = set(self.data.training().target_data())
        test_class_set = set(self.data.testing().target_data())
        if not test_class_set <= class_set:
            self.log.error("There are test classes %s that are not in the training class set %s",
                           ",".join(list(test_class_set)), ",".join(list(class_set)))
            self.log.fatal("OSM_QSAR cannot continue.")
            sys.exit()
        class_list = list(class_set)  # convert back to set
        sorted_class_list = sorted(class_list)   # ascending sort.
        return sorted_class_list

    def model_classification_results(self):
        self.train_predictions = self.model_prediction(self.data.training())  # Returns a dict. with "prediction" and "actual"
        self.train_probability = self.model_probability(self.data.training())  # Returns a dict. with "probability"
        self.train_stats = self.model_accuracy(self.train_predictions, self.train_probability)  # dictionary of stats

        self.test_predictions = self.model_prediction(self.data.testing())  # Returns a dict. with "prediction" and "actual"
        self.test_probability = self.model_probability(self.data.testing())  # Returns a dict. with "probability"
        self.test_stats = self.model_accuracy(self.test_predictions, self.test_probability)  # dictionary of stats
        # Send statistics to the console and log file.
        self.model_log_statistics()
        # Generate graphics (only if the virtual function defined at model level).
        self.model_graphics()
        # Append statistics to the stats file.
        self.model_write_statistics()

    def model_log_statistics(self):
        self.log_train_statistics(self.data.training(),
                                  self.train_stats,
                                  self.train_predictions,
                                  self.train_probability)
        self.log_test_statistics(self.data.testing(),
                                 self.test_stats,
                                 self.test_predictions,
                                 self.test_probability)

    def model_write_statistics(self):
        self.write_statistics(self.data.training(),
                              self.train_stats,
                              self.train_predictions,
                              self.train_probability,
                              self.args.trainDirectory)
        self.write_statistics(self.data.testing(),
                              self.test_stats,
                              self.test_predictions,
                              self.test_probability,
                              self.args.testDirectory)

    def model_accuracy(self, predictions, probability):
        predict = predictions["prediction"]
        actual = predictions["actual"]
        probabilities = probability["probability"]
        inv_probability = [1 - x[0] for x in probabilities]  # only interested in the first column ("active")
        # Sort rankings.
        probability_ranks = st.rankdata(inv_probability, method="average")

        #        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        #        auc_stat = auc(predict, actual)
        auc_stat = 0

        actual_text = OSMUtility.one_hot_text(actual, self.model_enumerate_classes())
        predict_text = OSMUtility.probability_text(probabilities, self.model_enumerate_classes())

        # Return the model analysis statistics in a dictionary.
        return {"AUC": auc_stat, "actual": actual, "predict": predict, "prob_rank": probability_ranks,
                "actual_text": actual_text, "predict_text": predict_text}

    #####################################################################################
    #
    # Local member functions
    #
    #####################################################################################

    def log_train_statistics(self, data, statistics, predictions, probabilities):

        self.log.info("Training Compounds Area Under Curve (AUC): %f", statistics["AUC"])

    # Display the classification results and write to the log file.
    def log_test_statistics(self, data, statistics, predictions, probabilities):
        """Display all the calculated statistics for each model; run"""

        self.log.info("Test Compounds Area Under Curve: %f", statistics["AUC"])
        self.log.info("ID, Actual Class, Pred. Class, Prob. Active, Prob. Rank")
        self.log.info("===================================================================================")

        for idx in range(len(data.get_field("ID"))):
            self.log.info("%s, %s, %s, %f, %d", data.get_field("ID")[idx],
                          self.test_stats["actual_text"][idx],
                          self.test_stats["predict_text"][idx],
                          self.test_probability["probability"][idx][0],
                          self.test_stats["prob_rank"][idx])

    # Open the statistics file and append the model results statistics.
    def write_statistics(self, data, statistics, predictions, probabilities, directory):

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
                line = "++++++++++++++++,Test Statistics,++++++++++++++++\n"
                stats_file.write(line)
                line = "AUC, {}\n".format(statistics["AUC"])
                stats_file.write(line)
                line = "ID, Actual_Class, Pred_Class "
                classes = self.model_enumerate_classes()
                for cls in classes:
                    line += ", Prob_" + cls
                line += ", SMILE\n"
                stats_file.write(line)

                for idx in range(len(data.get_field("ID"))):
                    line = "{}, {}, {}".format(data.get_field("ID")[idx],
                                               statistics["actual_text"][idx],
                                               statistics["predict_text"][idx])

                    for cls_idx in range(probabilities["probability"][idx].size):
                        line += ", {}".format(probabilities["probability"][idx][cls_idx])

                    line += ", {}\n".format(data.get_field("SMILE")[idx])

                    stats_file.write(line)

        except IOError:
            self.log.error("Problem writing to statistics file %s, check path and permissions", stats_filename)
