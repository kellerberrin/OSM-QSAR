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
# Python 2 and Python 3 compatibility imports.
from __future__ import absolute_import, division, print_function, unicode_literals
from six import with_metaclass


import numpy as np
from sklearn.preprocessing import label_binarize

import tensorflow as tf

from OSMBase import ModelMetaClass  # The virtual model class.
from OSMClassify import OSMClassification  # Display and save regression results.
from OSMGraphics import OSMSimilarityMap


# ===============================================================================
# Base class for the Keras neural network classifiers.
# ===============================================================================

class TensorFlowClassifier(OSMClassification):

    def __init__(self, args, log):
        super(TensorFlowClassifier, self).__init__(args, log)

    # ===============================================================================
    # Base class for the Keras neural network classifiers.
    # ===============================================================================

    def model_read(self, file_name):
        self.log.info("TENSORFLOW - Loading Pre-Trained %s Model File: %s", self.model_name(), file_name)
        return self.model_define()

    def model_write(self, file_name):
        self.log.info("TENSORFLOW - Saving Trained %s Model in File: %s", self.model_name(), file_name)
        saver = tf.train.Saver()
        saver.save(self.model["sess"], file_name)


    def model_graphics(self):


        def tensorflow_probability(fp, model):
            int_list = []
            float_list =[]
            for arr in fp:
                int_list.append(arr)

            shape = [int_list]
            fp_floats = np.array(shape, dtype=float)
            probs = model["sess"].run(model["ph_probability"], feed_dict={model["ph_input"]: fp_floats})
#            active_prob = prob_func(fp_floats)[0][0]  # returns an "active" probability (element[0]).
            return probs[0][0]

        func = lambda x: tensorflow_probability(x, self.model)


        OSMSimilarityMap(self, self.data.testing(), func).maps(self.args.testDirectory)
        if self.args.extendFlag:
            OSMSimilarityMap(self, self.data.training(), func).maps(self.args.trainDirectory)


# ===============================================================================
# The sequential neural net class developed by Vito Spadavecchio.
# ===============================================================================

class TensorFlowSimpleClassifier(with_metaclass(ModelMetaClass, TensorFlowClassifier)):

    def __init__(self, args, log):
        super(TensorFlowSimpleClassifier, self).__init__(args, log)

        # define the model data view.
        self.arguments = { "DEPENDENT" : { "VARIABLE" : "IC50_ACTIVITY", "SHAPE" : [1], "TYPE": np.str }
                         , "INDEPENDENT" : [ { "VARIABLE" : "MORGAN2048", "SHAPE": [2048], "TYPE": np.float64 } ] }

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "Simple TENSORFLOW Classifier"

    def model_postfix(self):  # Must be unique for each model.
        return "tfsim"

    def model_description(self):
        return ("A simple TENSORFLOW based Neural Network classifier based on the example simple MNIST classifier.\n"
                "See extensive documentation at: http://tensorflow.org/tutorials/mnist/beginners/index.md\n"
                "The classifier uses 2048 bit Morgan molecular fingerprints in a single layer NN.")

    def model_define(self):

        # Create the model
        ph_input = tf.placeholder(tf.float32, [None, 2048])
        W = tf.Variable(tf.zeros([2048, 3]))
        b = tf.Variable(tf.zeros([3]))
        ph_predict = tf.matmul(ph_input, W) + b

        # Define loss and optimizer
        ph_target = tf.placeholder(tf.float32, [None, 3])
        # Define the probability function.
        ph_probability = tf.nn.softmax(ph_predict)

        # The raw formulation of cross-entropy,
        #
        #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
        #                                 reduction_indices=[1]))
        #
        # can be numerically unstable.
        #
        # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
        # outputs of 'y', and then average across the batch.
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ph_target, logits=ph_predict))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        sess = tf.Session()
        tf.global_variables_initializer().run(session=sess)

        model = {"sess" : sess, "train_step" : train_step, "ph_input" : ph_input,
                "ph_target" : ph_target , "ph_predict" : ph_predict, "ph_probability": ph_probability}

        return model

    def model_prediction(self, data):

        classes = self.model_enumerate_classes()

        sess = self.model["sess"]
        ph_predict = self.model["ph_predict"]
        ph_input = self.model["ph_input"]

        predict_class_index = sess.run(tf.argmax(ph_predict, 1), feed_dict={ph_input : data.input_data()})
        predicted_classes = []
        for prediction in predict_class_index:
            predicted_classes.append(classes[prediction])

        return {"prediction": predicted_classes, "actual": data.target_data()}

    def model_train(self):

        classes = self.model_enumerate_classes()
        # Train
        sess = self.model["sess"]
        train_step = self.model["train_step"]
        ph_target = self.model["ph_target"]
        ph_input = self.model["ph_input"]

        train_one_hot = label_binarize(self.data.training().target_data(), classes)

        for _ in range(1000):
            sess.run(train_step, feed_dict={ph_input: self.data.training().input_data(), ph_target: train_one_hot})

    def model_probability(self, data):  # probabilities are returned as a numpy.shape = (samples, classes)

        sess = self.model["sess"]
        ph_input = self.model["ph_input"]
        ph_probability = self.model["ph_probability"]
        probability = sess.run(ph_probability, feed_dict={ph_input : data.input_data()})
        return {"probability": probability}

