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
import matplotlib.pyplot as plt


import tensorflow as tf

from OSMBase import ModelMetaClass, OSMBaseModel  # The virtual model class.
from OSMClassify import OSMClassification  # Display and save regression results.
from OSMModelData import OSMModelData
from OSMGraphics import OSMSimilarityMap
from OSMIterative import OSMIterative


# ===============================================================================
# Base class for Tensor flow neural network classifiers.
# ===============================================================================

class TensorFlowRNN(OSMBaseModel):

    def __init__(self, args, log):
        super(TensorFlowRNN, self).__init__(args, log)


    def init_parameters(self):
        self.num_epochs = 100
        self.total_series_length = 50000
        self.truncated_backprop_length = 15
        self.state_size = 4
        self.num_classes = 2
        self.echo_step = 3
        self.batch_size = 5
        self.num_batches = self.total_series_length//self.batch_size//self.truncated_backprop_length

    def generateData(self):
        x = np.array(np.random.choice(2, self.total_series_length, p=[0.5, 0.5]))
        y = np.roll(x, self.echo_step)
        y[0:self.echo_step] = 0

        x = x.reshape((self.batch_size, -1))  # The first index changing slowest, subseries as rows
        y = y.reshape((self.batch_size, -1))
        return (x, y)

    def tf_setup(self):

        self.batchX_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.truncated_backprop_length])
        self.batchY_placeholder = tf.placeholder(tf.int32, [self.batch_size, self.truncated_backprop_length])

        self.init_state = tf.placeholder(tf.float32, [self.batch_size, self.state_size])

        W = tf.Variable(np.random.rand(self.state_size+1, self.state_size), dtype=tf.float32)
        b = tf.Variable(np.zeros((1,self.state_size)), dtype=tf.float32)

        W2 = tf.Variable(np.random.rand(self.state_size, self.num_classes),dtype=tf.float32)
        b2 = tf.Variable(np.zeros((1,self.num_classes)), dtype=tf.float32)

        # Unpack columns
        inputs_series = tf.unpack(self.batchX_placeholder, axis=1)
        labels_series = tf.unpack(self.batchY_placeholder, axis=1)

        # Forward pass
        self.current_state = self.init_state
        states_series = []
        for current_input in inputs_series:
            current_input = tf.reshape(current_input, [self.batch_size, 1])
            input_and_state_concatenated = tf.concat(1, [current_input, self.current_state])  # Increasing number of columns

            next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  # Broadcasted addition
            states_series.append(next_state)
            self.current_state = next_state

        logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
        self.predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

        losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) for logits, labels in zip(logits_series,labels_series)]
        self.total_loss = tf.reduce_mean(losses)

        self.train_step = tf.train.AdagradOptimizer(0.3).minimize(self.total_loss)

    def plot(self, loss_list, predictions_series, batchX, batchY):
        plt.subplot(2, 3, 1)
        plt.cla()
        plt.plot(loss_list)

        for batch_series_idx in range(5):
            one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
            single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

            plt.subplot(2, 3, batch_series_idx + 2)
            plt.cla()
            plt.axis([0, self.truncated_backprop_length, 0, 2])
            left_offset = range(self.truncated_backprop_length)
            plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
            plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
            plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

        plt.draw()
        plt.pause(0.0001)


    def tf_run(self):

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            plt.ion()
            plt.figure()
            plt.show()
            loss_list = []

            for epoch_idx in range(self.num_epochs):
                x,y = self.generateData()
                _current_state = np.zeros((self.batch_size, self.state_size))

                print("New data, epoch", epoch_idx)

                for batch_idx in range(self.num_batches):
                    start_idx = batch_idx * self.truncated_backprop_length
                    end_idx = start_idx + self.truncated_backprop_length

                    batchX = x[:,start_idx:end_idx]
                    batchY = y[:,start_idx:end_idx]

                    _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                        [self.total_loss, self.train_step, self.current_state, self.predictions_series],
                        feed_dict={ self.batchX_placeholder:batchX,
                                    self.batchY_placeholder:batchY,
                                    self.init_state:_current_state})

                    loss_list.append(_total_loss)

                    if batch_idx%100 == 0:
                        print("Step",batch_idx, "Loss", _total_loss)
                        self.plot(loss_list, _predictions_series, batchX, batchY)

            plt.ioff()
            plt.show()