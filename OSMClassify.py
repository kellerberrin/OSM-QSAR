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

import numpy as np
import scipy.stats as st
from sklearn.metrics import auc, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize

from OSMBase import OSMBaseModel
from OSMModelData import OSMModelData

# ============================================================================
# The Classification Results Presentation Object.
# ============================================================================


class OSMClassification(OSMBaseModel):
    def __init__(self, args, log):
        super(OSMClassification, self).__init__(args, log)

        # Shallow copies of the runtime environment.
        self.log = log
        self.args = args

        # Maintain a vector of statistics

        self.test_statistics_history = []
        self.train_statistics_history = []

    #####################################################################################
    #
    # Virtual member functions called from OSMBase
    #
    #####################################################################################

    def model_is_classifier(self):
        return True


    def model_classification_results(self):
        self.train_predictions = self.model_prediction(self.data.training())  # Returns a dict. with "prediction" and "actual"
        self.train_probability = self.model_probability(self.data.training())  # Returns a dict. with "probability"
        self.train_objective = self.model_evaluate(self.data.training())
        self.train_stats = self.model_accuracy(self.train_predictions, self.train_probability)  # dictionary of stats
        self.train_statistics_history.append({"STATISTICS": self.train_stats, "OBJECTIVE": self.train_objective})

        self.test_predictions = self.model_prediction(self.data.testing())  # Returns a dict. with "prediction" and "actual"
        self.test_probability = self.model_probability(self.data.testing())  # Returns a dict. with "probability"
        self.test_objective = self.model_evaluate(self.data.testing())
        self.test_stats = self.model_accuracy(self.test_predictions, self.test_probability)  # dictionary of stats
        self.test_statistics_history.append({"STATISTICS": self.test_stats, "OBJECTIVE": self.test_objective})

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
                                  self.train_probability,
                                  self.train_objective)
        self.log_test_statistics(self.data.testing(),
                                 self.test_stats,
                                 self.test_predictions,
                                 self.test_probability,
                                 self.test_objective)

    def model_write_statistics(self):
        self.write_statistics(self.data.training(),
                              self.train_stats,
                              self.train_predictions,
                              self.train_probability,
                              self.train_objective,
                              self.args.trainDirectory)
        self.write_statistics(self.data.testing(),
                              self.test_stats,
                              self.test_predictions,
                              self.test_probability,
                              self.test_objective,
                              self.args.testDirectory)

    def model_training_summary(self):
        self.write_training_statistics(self.train_statistics_history,self.args.trainDirectory)
        self.write_training_statistics(self.test_statistics_history,self.args.testDirectory)
        self.write_model_analytics(self.model_analytics(), self.args.testDirectory)

    def model_accuracy(self, predictions, probability):

        classes = self.model_enumerate_classes()
        predict_text = predictions["prediction"]
        actual_text = predictions["actual"]
        probabilities = probability["probability"]
        inv_probability = [1 - x[0] for x in probabilities]  # only interested in the first column ("active")
        # Sort rankings.
        probability_ranks = st.rankdata(inv_probability, method="average")

        actual_one_hot =label_binarize(actual_text, classes)
        predict_one_hot = label_binarize(predict_text, classes)

        if len(classes) == 2 and actual_one_hot.shape[1] == 1:
            auc_probs = [ x[1] for x in probabilities]
#            print("auc_prob", auc_probs)
#            auc_probs = probabilities[:,1]
        else:
            auc_probs = probabilities

        class_auc = roc_auc_score(actual_one_hot, auc_probs, average=None, sample_weight=None)
        macro_auc = roc_auc_score(actual_one_hot, auc_probs, average="macro", sample_weight=None)
        micro_auc = roc_auc_score(actual_one_hot, auc_probs, average="micro", sample_weight=None)

        if len(classes) == 2 and actual_one_hot.shape[1] == 1:
            mod_class_auc = None
        else:
            mod_class_auc = class_auc

        confusion = confusion_matrix(actual_text, predict_text)

        epoch = self.model_epochs()

        # Return the model analysis statistics in a dictionary.
        return {"classes": classes, "actual_one_hot": actual_one_hot, "predict_one_host": predict_one_hot
               , "prob_rank": probability_ranks, "actual_text": actual_text, "predict_text": predict_text
               , "confusion" : confusion, "class_auc": mod_class_auc, "macro_auc": macro_auc
               , "micro_auc": micro_auc, "epoch": epoch }


    def model_prediction_records(self, data, statistics, predictions, probabilities):

        classes = self.model_enumerate_classes()
        prediction_list = []
        for idx in range(len(data.get_field("ID"))):
            prediction_record = []
            prediction_record.append(data.get_field("ID")[idx])
            prediction_record.append(statistics["actual_text"][idx])
            prediction_record.append(statistics["predict_text"][idx])
            prediction_record.append(statistics["prob_rank"][idx])
            prediction_record.append(data.get_field("SMILE")[idx])
            prob_list = []
            for cls_idx in range(len(classes)):
                prob_list.append(probabilities["probability"][idx][cls_idx])
            prediction_record.append(prob_list)

            prediction_list.append(prediction_record)

        # Sort by actual ranking (inverse order).
        sorted_predict_list= sorted(prediction_list, key=lambda predict_record: (predict_record[5][0] * -1))

        return sorted_predict_list


    #####################################################################################
    #
    # Local member functions
    #
    #####################################################################################

    def log_train_statistics(self, data, statistics, predictions, probabilities, objective):

        self.log.info("Training Compounds macro AUC: %f", statistics["macro_auc"])
        self.log.info("Training Compounds micro AUC: %f", statistics["micro_auc"])
        if statistics["class_auc"] is not None:
            for aclass, auc in zip(statistics["classes"], statistics["class_auc"]):
                self.log.info("Training Compounds Class %s AUC: %f", aclass, auc)
        for idx in range(len(objective)):
            self.log.info("Train Model Objective-%d, %f",idx, objective[idx])

    # Display the classification results and write to the log file.
    def log_test_statistics(self, data, statistics, predictions, probabilities, objective):
        """Display all the calculated statistics for each model; run"""
        independent_list = []
        for var in self.model_arguments()["INDEPENDENT"]:
            independent_list.append(var["VARIABLE"])
        dependent_var = self.model_arguments()["DEPENDENT"]["VARIABLE"]

        self.log.info("Dependent (Target) Variable: %s", dependent_var)
        for var in independent_list:
            self.log.info("Independent (Input) Variable(s): %s", var)
        self.log.info("Training Epochs: %d", self.model_epochs())
        for idx in range(len(objective)):
            self.log.info("Test Model Objective-%d, %f",idx, objective[idx])
        self.log.info("Test Compounds macro AUC: %f", statistics["macro_auc"])
        self.log.info("Test Compounds micro AUC: %f", statistics["micro_auc"])
        if statistics["class_auc"] is not None:
            for aclass, auc in zip(statistics["classes"], statistics["class_auc"]):
                self.log.info("Test Class %s AUC: %f", aclass, auc)
        self.log.info("+++++++++++++++ Confusion matrix +++++++++++++++++++++++")
        line = "true/predict  "
        for a_class in statistics["classes"]:
            line += "{:10s}".format(a_class)
        self.log.info(line)
        for rowidx in range(len(statistics["confusion"])):
            line = "{:10s}".format(statistics["classes"][rowidx])
            for colidx in range(len(statistics["confusion"][rowidx])):
                line += "{:8d}".format(statistics["confusion"][rowidx][colidx])
            self.log.info(line)
        self.log.info("ID, Actual Class, Pred. Class, Prob. Rank, Prob. %s", statistics["classes"][0])
        self.log.info("===================================================================================")

        sorted_records = self.model_prediction_records(data, statistics, predictions, probabilities)

        for record in sorted_records:
            line = "{:10s} {:10s} {:10s} {:3.1f}   {:8.7f}".format(record[0], record[1],
                                                                   record[2], record[3], record[5][0])
            self.log.info(line)

    # Open the statistics file and append the model results statistics.
    def write_statistics(self, data, statistics, predictions, probabilities, objective, directory):

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
                for idx in range(len(objective)):
                    line = "ModelObjective-{}, {}\n".format(idx,objective[idx])
                    stats_file.write(line)
                line = "++++++++++++++++,Test_Statistics,++++++++++++++++\n"
                stats_file.write(line)
                line = "Macro AUC, {}\n".format(statistics["macro_auc"])
                stats_file.write(line)
                line = "Micro AUC, {}\n".format(statistics["micro_auc"])
                stats_file.write(line)
                if statistics["class_auc"] is not None:
                    for aclass, auc in zip(statistics["classes"], statistics["class_auc"]):
                        line = "Class {} AUC, {}\n".format(aclass, auc)
                        stats_file.write(line)
                stats_file.write("Confusion matrix\n")
                line = "true/predict"
                for a_class in statistics["classes"]:
                    line += ",{}".format(a_class)
                line += "\n"
                stats_file.write(line)
                for rowidx in range(len(statistics["confusion"])):
                    line = "{}".format(statistics["classes"][rowidx])
                    for colidx in range(len(statistics["confusion"][rowidx])):
                        line += ",{}".format(statistics["confusion"][rowidx][colidx])
                    line += "\n"
                    stats_file.write(line)

                sorted_records = self.model_prediction_records(data, statistics, predictions, probabilities)

                line = "++++++++++++++++,Compound_Statistics,++++++++++++++++\n"
                stats_file.write(line)
                line = "ID, Actual_Class, Pred_Class"
                for cls in statistics["classes"]:
                    line += ", Prob_" + cls
                line += ", SMILE\n"
                stats_file.write(line)

                for record in sorted_records:
                    line = "{}, {}, {}".format(record[0], record[1], record[2])

                    for cls_idx in range(len(statistics["classes"])):
                        line += ", {}".format(record[5][cls_idx])

                    line += ", {}\n".format(record[4])

                    stats_file.write(line)

        except IOError:
            self.log.error("Problem writing to statistics file %s, check path and permissions", stats_filename)

    def write_training_statistics(self, statistics_vector, directory):

        stats_filename = os.path.join(directory, self.args.statsFilename)

        try:

            with open(stats_filename, 'a') as stats_file:


                line = "++++++++++++++++++,Training_Summary,+++++++++++++++++++\n"
                stats_file.write(line)

                for statistics in statistics_vector:
                    micro_AUC = statistics["STATISTICS"]["micro_auc"]
                    macro_AUC = statistics["STATISTICS"]["macro_auc"]
                    epoch = statistics["STATISTICS"]["epoch"]
                    line = "epoch, {}, micro AUC, {}, macro AUC, {}".format(epoch, micro_AUC, macro_AUC)
                    objective = statistics["OBJECTIVE"]
                    for idx in range(len(objective)):
                        line += ",Objective-{}, {}".format(idx, objective[idx])
                    stats_file.write(line+"\n")

        except IOError:
            self.log.error("Problem writing to statistics file %s, check path and permissions", stats_filename)


    def write_model_analytics(self, sensitivity_dict, directory):

        if sensitivity_dict is None: return

        if "SENSITIVITY" in sensitivity_dict:
            self.write_analytic_type(directory, sensitivity_dict["SENSITIVITY"], "SENSITIVITY")

        if "DERIVATIVE" in sensitivity_dict:
            self.write_analytic_type(directory, sensitivity_dict["DERIVATIVE"], "DERIVATIVE")

    def write_analytic_type(self, directory, sensitivity_vector, title):

        stats_filename = os.path.join(directory, self.args.statsFilename)

        try:

            with open(stats_filename, 'a') as stats_file:


                line = "++++++++++++++++++,{},+++++++++++++++++++\n".format(title)
                stats_file.write(line)

                for field_sens in sensitivity_vector:
                    line = "field, {}, index, {}, sensitivity, {}\n".format(field_sens[0], field_sens[1], field_sens[2])
                    stats_file.write(line)

        except IOError:
            self.log.error("Problem writing to sensitivity file %s, check path and permissions", stats_filename)

    def model_enumerate_classes(self):

        training_classes = OSMModelData.enumerate_classes(self.data.training().target_data())
        test_classes = OSMModelData.enumerate_classes(self.data.testing().target_data())
        if not set(test_classes) <= set(training_classes):
            self.log.error("There are test classes %s not in the set of training classes %s",
                           ",".join(test_classes), ",".join(training_classes))
            sys.exit()
        return training_classes
