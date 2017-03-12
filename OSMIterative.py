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

import sys


# ===============================================================================
# Implements iterative model functionality.
# ===============================================================================

class OSMIterative(object):

    def __init__(self, iterative_model):

        self.iterative_model = iterative_model
        self.default_epochs = 1000
        self.__trained_epochs__ = 0
        self.__total_epochs__ = 0

    def read(self):

        if self.iterative_model.args.epoch > 0:
            self.__trained_epochs__ = self.iterative_model.args.epoch
            self.__total_epochs__ = self.iterative_model.args.epoch
        else:
            self.iterative_model.log.error('ITERATIVE - Flag "--epoch" must be specified with  "--load" see "--help"')
            sys.exit()

        try:
            model = self.iterative_model.epoch_read(self.__trained_epochs__)
        except IOError:
            self.iterative_model.log.error('ITERATIVE - Model %s Failed to load, "--load %s" "--epoch %d", check flags.'
                           , self.iterative_model.model_name(), self.iterative_model.args.loadFilename,
                                           self.iterative_model.args.epoch)
            sys.exit()

        if model is None:
            self.iterative_model.log.error('ITERATIVE - Model %s Failed to load, "--load %s" "--epoch %d", check flags.'
                           , self.iterative_model.model_name(), self.iterative_model.args.loadFilename
                           , self.iterative_model.args.epoch)
            sys.exit()

        return model

    def write(self):
        self.iterative_model.epoch_write(self.__trained_epochs__)

    def train(self, default_epochs):

        if 0 < self.iterative_model.args.checkPoint < self.iterative_model.args.train:

            self.__total_epochs__ += self.iterative_model.args.train

            while self.__trained_epochs__ < self.__total_epochs__:
                if (self.__total_epochs__ - self.__trained_epochs__) > self.iterative_model.args.checkPoint:
                    train_epochs = self.iterative_model.args.checkPoint
                else:
                    train_epochs = self.__total_epochs__ - self.__trained_epochs__

                self.iterative_model.log.info("ITERATIVE - Begin Training %s Model %d epochs"
                              , self.iterative_model.model_name(), train_epochs)
                self.iterative_model.train_epoch(train_epochs)
                self.iterative_model.log.info("ITERATIVE - End Training %s Model", self.iterative_model.model_name())

                self.__trained_epochs__ += train_epochs
                if self.__trained_epochs__ < self.__total_epochs__:
                    self.iterative_model.epoch_write(self.__trained_epochs__)
                    self.iterative_model.model_classification_results()

        elif self.iterative_model.args.train > 0:
            self.__total_epochs__ += self.iterative_model.args.train
            self.iterative_model.train_epoch(self.iterative_model.args.train)
            self.__trained_epochs__ += self.iterative_model.args.train

        elif self.iterative_model.args.train < 0:    # the train flag has not been specified.
            self.__total_epochs__ += default_epochs
            self.iterative_model.train_epoch(default_epochs)
            self.__trained_epochs__ += default_epochs

    def trained_epochs(self):
        return self.__trained_epochs__