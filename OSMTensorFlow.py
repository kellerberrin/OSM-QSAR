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
import os

import numpy as np
from sklearn.preprocessing import label_binarize

import tensorflow as tf

from OSMBase import ModelMetaClass  # The virtual model class.
from OSMClassify import OSMClassification  # Display and save regression results.
from OSMModelData import OSMModelData
from OSMGraphics import OSMSimilarityMap
from OSMIterative import OSMIterative


# ===============================================================================
# Base class for Tensor flow neural network classifiers.
# ===============================================================================

class TensorFlowClassifier(OSMClassification):

    def __init__(self, args, log):
        super(TensorFlowClassifier, self).__init__(args, log)

        self.default_epochs = 1000

        self.iterative = OSMIterative(self)


# ===============================================================================
# Base class for Tensor flow neural network classifiers.
# ===============================================================================

    def model_write(self):
        self.iterative.write()

    def model_read(self):
        return self.iterative.read()

    def model_train(self):
        self.iterative.train(self.default_epochs)

    def model_epochs(self):
        return self.iterative.trained_epochs()

    def epoch_read(self, epoch):
        self.log.info("TENSORFLOW - Loading Pre-Trained %s Model File: %s, epoch: %d"
                      , self.model_name(), self.args.loadFilename, epoch)
        file_name = self.args.loadFilename + "-" + "{}".format(epoch)
        model = self.model_define()
        saver = tf.train.Saver()
        saver.restore(model["sess"], file_name)
        return model

    def epoch_write(self, epoch):
        self.log.info("TENSORFLOW - Saving Trained %s Model in File: %s, epoch %d"
                      , self.model_name(), self.args.saveFilename, epoch)
        saver = tf.train.Saver()
        saver.save(self.model["sess"], self.args.saveFilename, global_step=epoch)

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

        if self.args.checkPoint < 0 or self.args.extendFlag:
            OSMSimilarityMap(self, self.data.testing(), func).maps(self.args.testDirectory)
            if self.args.extendFlag:
                OSMSimilarityMap(self, self.data.training(), func).maps(self.args.trainDirectory)


# ===============================================================================
# A simple tensorflow example
# ===============================================================================

class TensorFlowSimpleClassifier(with_metaclass(ModelMetaClass, TensorFlowClassifier)):

    def __init__(self, args, log):
        super(TensorFlowSimpleClassifier, self).__init__(args, log)

        # define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = { "DEPENDENT" : { "VARIABLE" : "ION_ACTIVITY", "SHAPE" : [3], "TYPE": OSMModelData.CLASSES }
                    , "INDEPENDENT" : [ { "VARIABLE" : "MORGAN2048_4", "SHAPE": [2048], "TYPE": OSMModelData.FLOAT64 } ] }

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
        W = tf.Variable(tf.zeros([2048, 3]), name="Weights")
        b = tf.Variable(tf.zeros([3]), name="Bias")
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
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

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

    def train_epoch(self, epoch):

        classes = self.model_enumerate_classes()
        # Train
        sess = self.model["sess"]
        train_step = self.model["train_step"]
        ph_target = self.model["ph_target"]
        ph_input = self.model["ph_input"]

        train_one_hot = label_binarize(self.data.training().target_data(), classes)

        for _ in range(epoch):
            sess.run(train_step, feed_dict={ph_input: self.data.training().input_data(), ph_target: train_one_hot})


    def model_probability(self, data):  # probabilities are returned as a numpy.shape = (samples, classes)

        sess = self.model["sess"]
        ph_input = self.model["ph_input"]
        ph_probability = self.model["ph_probability"]
        probability = sess.run(ph_probability, feed_dict={ph_input : data.input_data()})
        return {"probability": probability}



# ===============================================================================
# A simple tensorflow example
# ===============================================================================

class DTensorFlowSimpleClassifier(with_metaclass(ModelMetaClass, TensorFlowClassifier)):

    def __init__(self, args, log):
        super(DTensorFlowSimpleClassifier, self).__init__(args, log)

        # define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = { "DEPENDENT" : { "VARIABLE" : "ION_ACTIVITY", "SHAPE" : [3], "TYPE": OSMModelData.CLASSES }
                    , "INDEPENDENT" : [ { "VARIABLE" : "DRAGON", "SHAPE": [1552], "TYPE": OSMModelData.FLOAT64 } ] }

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "Simple TENSORFLOW Classifier"

    def model_postfix(self):  # Must be unique for each model.
        return "tfd"

    def model_description(self):
        return ("A simple TENSORFLOW based Neural Network classifier based on the example simple MNIST classifier.\n"
                "See extensive documentation at: http://tensorflow.org/tutorials/mnist/beginners/index.md\n")

    def model_define(self):

        # Create the model
        ph_input = tf.placeholder(tf.float32, [None, 1552])
        W = tf.Variable(tf.zeros([1552, 3]), name="Weights")
        b = tf.Variable(tf.zeros([3]), name="Bias")
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
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

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

    def train_epoch(self, epoch):

        classes = self.model_enumerate_classes()
        # Train
        sess = self.model["sess"]
        train_step = self.model["train_step"]
        ph_target = self.model["ph_target"]
        ph_input = self.model["ph_input"]

        train_one_hot = label_binarize(self.data.training().target_data(), classes)

        for _ in range(epoch):
            sess.run(train_step, feed_dict={ph_input: self.data.training().input_data(), ph_target: train_one_hot})


    def model_probability(self, data):  # probabilities are returned as a numpy.shape = (samples, classes)

        sess = self.model["sess"]
        ph_input = self.model["ph_input"]
        ph_probability = self.model["ph_probability"]
        probability = sess.run(ph_probability, feed_dict={ph_input : data.input_data()})
        return {"probability": probability}


# ===============================================================================
# A tensorflow tf.contrib.learn DNN example
# ===============================================================================


class TF_DNN_Classifier(with_metaclass(ModelMetaClass, TensorFlowClassifier)):

    def __init__(self, args, log):
        super(TF_DNN_Classifier, self).__init__(args, log)

        # define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = { "DEPENDENT" : { "VARIABLE" : "ION_ACTIVITY", "SHAPE" : [2], "TYPE": OSMModelData.CLASSES }
                    , "INDEPENDENT" : [ { "VARIABLE" : "MORGAN2048_4", "SHAPE": [1552], "TYPE": OSMModelData.FLOAT64 } ] }

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "TENSORFLOW tf.contrib.learn DNNClassifier"

    def model_postfix(self):  # Must be unique for each model.
        return "tfdnn"

    def model_description(self):
        return ("A TENSORFLOW tf.learn based Neural Network classifier.\n"
                "See tf.learn documentation at: https://www.tensorflow.org/get_started/tflearn\n")

    def model_define(self):
        return self.model_define_directory(None)

    def model_define_directory(self, directory):
        # Specify that all features have real-value data
        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1552)]
        # Build 3 layer DNN.
        model = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                               hidden_units=[50, 50, 10],
                                               n_classes=2,
                                               model_dir = directory)
        return model

    def model_prediction(self, data):
        input_fn = lambda: self.input_data(data)
        predict = self.model.predict(input_fn=input_fn)
        predict_list = list(predict)
        classes = self.model_enumerate_classes()
        class_list = []
        for predict in predict_list:
            class_list.append(classes[predict])
        return {"prediction": class_list, "actual": data.target_data()}

    def epoch_read(self, epoch):
        postfix_directory = os.path.join(self.args.workDirectory, self.args.classifyType)
        postfix_directory += "/{0:0>8}".format(epoch)
        self.log.info("TENSORFLOW - Loading Pre-Trained %s Directory: %s, epoch: %d"
                      , self.model_name(), postfix_directory, epoch)

        return self.model_define_directory(postfix_directory)

    def epoch_write(self, epoch):
        postfix_directory = os.path.join(self.args.workDirectory, self.args.classifyType)
        postfix_directory_epoch = postfix_directory + "/{0:0>8}".format(epoch)
        self.log.info("TENSORFLOW - Saving Trained %s Model in Directory: %s, epoch %d"
                      , self.model_name(), postfix_directory_epoch, epoch)
        if not os.path.isdir(postfix_directory_epoch):
            self.model.export(export_dir=postfix_directory)
        else:
            self.log.warning("The model <postfix>/epoch directory: %s already exists. Cannot save model.",
                             postfix_directory_epoch)

    def train_epoch(self, epoch):

        input_fn = lambda: self.input_data(self.data.training())
        self.model.fit(input_fn=input_fn, steps=epoch)


    def model_probability(self, data):  # probabilities are returned as a numpy.shape = (samples, classes)
        input_fn = lambda: self.input_data(data)
        prob =self.model.predict_proba(input_fn=input_fn)
        list_prob = list(prob)
        prob_list = []
        for prob in list_prob:
          vec_list = []
          for p in prob:
            vec_list.append(p)
          prob_list.append(vec_list)
#        print("prob",prob_list)
        return {"probability": prob_list}

    def input_data(self, data):
        classes = self.model_enumerate_classes()
        class_list =  data.target_data()
        class_idx_list = []
        for a_class in class_list:
            class_idx_list.append(classes.index(a_class))
        input_y_tensor = tf.constant(class_idx_list)
        input_x_tensor = tf.constant(data.input_data())
        return input_x_tensor, input_y_tensor


# ===============================================================================
# A tensorflow tf.contrib.learn DNN example
# ===============================================================================


class DNN_MV_Classifier(with_metaclass(ModelMetaClass, TensorFlowClassifier)):

    def __init__(self, args, log):
        super(DNN_MV_Classifier, self).__init__(args, log)

        # define the model data view.
        # Define the model variable types here. Documented in "OSMModelData.py".
        self.arguments = { "DEPENDENT" : { "VARIABLE" : "ION_ACTIVITY", "SHAPE" : [3], "TYPE": OSMModelData.CLASSES }
                    , "INDEPENDENT" : [ { "VARIABLE" : "DRAGON", "SHAPE": [1666], "TYPE": OSMModelData.FLOAT64 }
                                    , { "VARIABLE" : "MORGAN2048_5", "SHAPE": [2048], "TYPE": OSMModelData.FLOAT64 }
                                    , {"VARIABLE": "MORGAN2048_1", "SHAPE": [2048], "TYPE": OSMModelData.FLOAT64}
                                    , {"VARIABLE": "TOPOLOGICAL2048", "SHAPE": [2048], "TYPE": OSMModelData.FLOAT64}
                                    , {"VARIABLE": "MACCFP", "SHAPE": [167], "TYPE": OSMModelData.FLOAT64}]}

    # These functions need to be re-defined in all classifier model classes.

    def model_name(self):
        return "Multi vector TENSORFLOW tf.contrib.learn DNNClassifier"

    def model_postfix(self):  # Must be unique for each model.
        return "dnnmv"

    def model_description(self):
        return ("A Multi Vector Neural Network classifier.\n"
                "Joins large input vectors, typically [Dragon, Fingerprint1, ....]\n")

    def model_define(self):
        return self.model_define_directory(None)

    def model_define_directory(self, directory):
        # Specify that all features have real-value data
        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=(1666))]
        # Build 3 layer DNN.
        model = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                               hidden_units=[2048, 2048, 16],
                                               n_classes=3,
                                               model_dir = directory)
        return model

    def model_prediction(self, data):
        input_fn = lambda: self.input_data(data)
        predict = self.model.predict(input_fn=input_fn)
        predict_list = list(predict)
        classes = self.model_enumerate_classes()
        class_list = []
        for predict in predict_list:
            class_list.append(classes[predict])
        return {"prediction": class_list, "actual": data.target_data()}

    def epoch_read(self, epoch):
        postfix_directory = os.path.join(self.args.workDirectory, self.args.classifyType)
        postfix_directory += "/{0:0>8}".format(epoch)
        self.log.info("TENSORFLOW - Loading Pre-Trained %s Directory: %s, epoch: %d"
                      , self.model_name(), postfix_directory, epoch)

        return self.model_define_directory(postfix_directory)

    def epoch_write(self, epoch):
        postfix_directory = os.path.join(self.args.workDirectory, self.args.classifyType)
        postfix_directory_epoch = postfix_directory + "/{0:0>8}".format(epoch)
        self.log.info("TENSORFLOW - Saving Trained %s Model in Directory: %s, epoch %d"
                      , self.model_name(), postfix_directory_epoch, epoch)
        if not os.path.isdir(postfix_directory_epoch):
            self.model.export(export_dir=postfix_directory)
        else:
            self.log.warning("The model <postfix>/epoch directory: %s already exists. Cannot save model.",
                             postfix_directory_epoch)

    def train_epoch(self, epoch):

        input_fn = lambda: self.input_data(self.data.training())
        self.model.fit(input_fn=input_fn, steps=epoch)


    def model_probability(self, data):  # probabilities are returned as a numpy.shape = (samples, classes)
        input_fn = lambda: self.input_data(data)
        prob =self.model.predict_proba(input_fn=input_fn)
        prob_list = list(prob)
        return {"probability": prob_list}

    def input_data(self, data):
        classes = self.model_enumerate_classes()
        class_list =  data.target_data()
        class_idx_list = []
        for a_class in class_list:
            class_idx_list.append(classes.index(a_class))
        input_y_tensor = tf.constant(class_idx_list)
        #        x_data = np.concatenate((data.input_data()["DRAGON"], data.input_data()["MORGAN2048_5"],
#            data.input_data()["MORGAN2048_1"],data.input_data()["TOPOLOGICAL2048"],data.input_data()["MACCFP"]), axis=1)
        input_x_tensor = tf.constant(data.input_data()["DRAGON"])
#        x_data = np.concatenate((data.input_data()["DRAGON"], data.input_data()["MACCFP"]), axis=1)
#        input_x_tensor = tf.constant(x_data)
        return input_x_tensor, input_y_tensor


