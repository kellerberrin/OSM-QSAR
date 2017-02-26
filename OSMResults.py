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


import math
import scipy.stats as st


# ============================================================================
# The Classification Results Presentation Object.
# ============================================================================


class OSMResults(object):


    def __init__(self, args, log):

        # Shallow copies of the runtime environment.
        self.log = log
        self.args = args


    def training_stats(self, model, train):
    
        self.trainPredictions = self.model_prediction(model, train)   # Returns a dict. with "prediction" and "actual"
        self.trainStats = self.model_accuracy(self.trainPredictions)  # Returns a dictionary of accuracy tests.
        self.log.info("Training Compounds pEC50 Mean Unsigned Error (MUE): %f", self.trainStats["MUE"])

    def model_accuracy(self, predictions):

        predict = predictions["prediction"]
        actual = predictions["actual"] 

        MUE = 0
        RMSE = 0
        for i in range(len(predict)):
            diff = abs(predict[i]-actual[i])
            MUE += diff
            RMSE = diff * diff            

        MUE = MUE / len(predict)
        RMSE = RMSE / len(predict)
        RMSE = math.sqrt(RMSE) 

# Sort rankings.

        predict_ranks = st.rankdata(predict, method='average')
        actual_ranks = st.rankdata(actual, method='average')

#  Active / Inactive classifications

        predict_active = self.classify_results(predict)
        actual_active = self.classify_results(actual)

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
        
        return {"MUE": MUE, "RMSE": RMSE,  "predict_ranks": predict_ranks,
                "actual_ranks": actual_ranks, "predict_active": predict_active,
                "actual_active": actual_active, "kendall" : kendall, "spearman" : spearman }

#  Active / Inactive args.activeNmols is an array of pEC50 potency sorted tuples [(potency, "classify"), ...]
    def classify_results(self, results):

        classified = []
        classes = self.args.activeNmols
        for x in results:
            if x <= classes[0][0]:
                class_str = classes[0][1]
            else:
                class_str = "inactive"

            if len(classes) > 1:
                for idx in range(len(classes)-1):
                    if x > classes[idx][0] and x <= classes[idx+1][0]:
                        class_str = classes[idx+1][1]
            classified.append(class_str)

        return classified

# Default for any model without similarity maps defined.
    def model_similarity(self, model, data): pass

# Display the classification results and write to the log file.
    def display_results(self, model, test):
        """Display all the calculated statistics for each model; run"""

        test_predictions = self.model_prediction(model, test)
        test_stats = self.model_accuracy(test_predictions)
                
        self.log.info("Test Compounds pEC50 Mean Unsigned Error (MUE): %f", test_stats["MUE"])
        self.log.info("Test Compounds Kendall's Rank Coefficient (tau): %f, p-value: %f",
                      test_stats["kendall"]["tau"], test_stats["kendall"]["p-value"])
        self.log.info("Test Compounds Spearman Coefficient (rho): %f, p-value: %f",
                      test_stats["spearman"]["rho"], test_stats["spearman"]["p-value"])
        self.log.info("Test Compounds pEC50 Mean Unsigned Error (MUE): %f", test_stats["MUE"])
        self.log.info("Test Compounds Results")
        self.log.info("ID, Tested Rank, Pred. Rank, Tested pEC50, Pred. pEC50, Tested Active, Pred. Active")
        self.log.info("===================================================================================")

        for idx in range(len(test["ID"])):
            self.log.info("%s, %d, %d, %f, %f, %s, %s", test["ID"][idx],
                                                        test_stats["actual_ranks"][idx],
                                                        test_stats["predict_ranks"][idx],
                                                        test_predictions["actual"][idx],
                                                        test_predictions["prediction"][idx],
                                                        test_stats["actual_active"][idx],
                                                        test_stats["predict_active"][idx]),

        self.model_similarity(model, test)

