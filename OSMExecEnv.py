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
import argparse
import logging
import time

# Import the various model objects
from OSMBase import get_model_instances
import OSMKeras
import OSMTemplate
import OSMSKLearnRegress
import OSMSKLearnClassify
import OSMTensorFlow

__version__ = "0.2"

# ===================================================================================================
# A utility class to parse the program runtime arguments
# and setup a logger to receive classification output.
# ===================================================================================================

class ExecEnv(object):
    """Utility class to setup the runtime environment and logging"""

    # Static class variables.

    args = None
    log = None
    cmdLine = ""
    modelInstances = []    # Model instances are singletons.

    def __init__(self):
        """Parse runtime arguments on object creation and maintain the runtime environment"""

        # Start a console logger to complain about any bad args (file logger defined below).

        file_log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ExecEnv.log = self.setup_logging(console_log_format)


        # Create the model instances - ***careful***, args are not yet defined.
        # Do not perform any classifications here.
        # The instances should only generate postfixes, descriptions and model names.
        # The model objects are held as singletons.

        ExecEnv.modelInstances = get_model_instances(ExecEnv.args, ExecEnv.log)

        # Generate a string of file extensions and create a suitable help string.

        model_postfix = ""
        for model_instance in ExecEnv.modelInstances:
            model_postfix += model_instance.model_postfix() + ", "

        classify_help = "Specify which classification model OSM_QSAR will execute using the model postfix code. "
        classify_help += "Current defined model postfixes are: "
        classify_help += model_postfix + '(default "seq").'
        classify_help += 'For more information on current models specify the "--model" flag.'

        # Parse the runtime args
        parser = argparse.ArgumentParser(
            description="OSM_QSAR. Classification of OSM ligands using machine learning techniques.")

        # --dir
        parser.add_argument("--dir", dest="workDirectory", default="./Work/",
                            help=('The work directory where log files and data files are found.'
                                  ' Use a Linux style directory specification with trailing forward'
                                  ' slash "/" (default "./Work/").'
                                  " Important - to run OSM_QSAR this directory must exist, it will not be created."
                                  ' Model specific files (models, statistics and graphics) are in the '
                                  'subdirectories "/<WorkDir>/postfix/...".'))

        # --data
        parser.add_argument("--data", dest="dataFilename", default="OSMData.csv",
                            help=('The input data filename (default "OSMData.csv").'
                                 " Important - to run OSM_QSAR this file must exist in the Work directory."
                                 ' For example, if this flag is not specified then OSM_QSAR attempts to read the'
                                 ' data file at "/<WorkDir>/OSMData.csv".'
                                 " See the additional OSM_QSAR documentation for the format of this file."))

        # --depend
        parser.add_argument("--depend", dest="dependVar", default="default",
                            help=('The regression or classifier dependent variable.'
                                 " This variable must exist in the data dictionary. The variables in the data"
                                 ' directory can be listed using the "--vars" flag. The default dependent variable'
                                 ' for a regression model is "pIC50" (log10 IC50 potency in uMols). The default'
                                 ' variable for a classifier model is "ION_ACTIVITY".'))

        # --indep
        parser.add_argument("--indep", dest="indepList", default="default",
                            help=('The regression or classifier independent variables.'
                                 " Important - some models (NNs) will not accept this flag and will issue a warning."
                                 " Specified variables must exist in the data dictionary. The variables in the data"
                                 ' directory can be listed using the "--vars" flag. The independent variables are '
                                 ' specified in a comma delimited list "Var1, Var2, ..., Varn". Quotes must present.'
                                 ' For regression and classifier models the default independent variable'
                                 ' is the Morgan molecular fingerprint "MORGAN2048".'))

        # --vars
        parser.add_argument("--vars", dest="varFlag", action="store_true",
                            help=("Lists all the data variables available in the data dictionary and exits."))
        # --load
        parser.add_argument("--load", dest="loadFilename", default="noload",
                            help=("Loads the saved model and generates statistics and graphics but does no "
                                  " further training."
                                  ' If the model is a Neural Network then the "--epoch" flag can be optionally'
                                  " specified to read to a particular NN epoch model."
                                  " This file is always located in the model <postfix> subdirectory of the Work"
                                  ' directory. For example, specifying "--load mymodel --epoch 1000 --train 0" for '
                                  ' a KERAS NN loads "./<WorkDir>/<postfix>/mymodel_1000.krs" and generates'
                                  ' model statistics and graphics without further training'))
        # --save
        parser.add_argument("--save", dest="saveFilename", default="OSMClassifier",
                            help=('File name to save the model (default "OSMClassifier").'
                                  ' The model file is always saved to the model postfix subdirectory.'
                                  ' Neural Networks append the training epoch to the file name.'
                                  ' For example, if a KERAS NN classifier is saved and if this flag is not specified '
                                  ' the model is saved to: "./<WorkDir>/postfix/OSMClassifier_<epoch>.krs".'
                                  ' The postfix directory is created if it does not exist.'))
        # --stats
        parser.add_argument("--stats", dest="statsFilename", default="OSMStatistics.csv",
                            help=('File to append the test and train model statistics (default "OSMStatistics").'
                                  ' The statistics files is always saved to subdirectories of the the model <postfix>'
                                  ' directory. Test data statistics are always appended to the specified statistics'
                                  ' file in the "./<WorkDir>/<postfix>/test/" and training data statistics are appended'
                                  ' to the specified statistics file in the "./<WorkDir>/<postfix>/train" directory.'
                                  ' The postfix directory and the "test" and "train" subdirectories are created'
                                  ' if they do not exist. The statistics file(s) are created if they do not exist.'))
        # --extend
        parser.add_argument("--extend", dest="extendFlag", action="store_true",
                            help=(' The "--extend" flag generates all training data statistics and graphics.'
                                  '  Additional training graphics and statistics are added to the'
                                  ' "./<WorkDir>/<postfix>/train" directory.'
                                  ' The directory is created is it does not exist. The statistics file is created'
                                  ' if it does not exist.'
                                  ' Warning - the "--extend" flag may substantially increase OSM_QSAR runtime.'))

        # --clean
        parser.add_argument("--clean", dest="cleanFlag", action="store_true",
                            help=('Deletes all files in the "test" and "train" subdirectories of the model '
                                  ' <postfix> directory before OSM_QSAR executes.'
                                  ' Any model files in the <postfix> directory are not deleted.'))
        # --log
        parser.add_argument("--log", dest="logFilename", default="OSM_QSAR.log",
                            help=('Log file. Appends the log to any existing logs (default "OSM_QSAR.log").'
                                  'The log file always resides in the work directory.'))
        # --newlog
        parser.add_argument("--newlog", dest="newLogFilename", default="nonewlog", nargs='?',
                            help='Flush an existing log file (file name argument optional, default "OSM_QSAR.log").'
                                 'The log file always resides in the work directory.')
        # --model
        parser.add_argument("--model", dest="modelDescriptions", action="store_true",
                            help=("Lists all defined regression and classification models and exits."))
        # --classify
        parser.add_argument("--classify", dest="classifyType", default="seq", help=classify_help)
        # --epoch
        parser.add_argument("--epoch", dest="epoch", default=-1, type=int,
                            help=(" Compulsory when loading Neural Networks and other iterative models."
                                  " Used to specify which training epoch to load'"
                                  " and retrain. Ignored if not valid for model.  Example:"
                                  '"--classify mod --load OSMClassifier -- epoch 1000 --train 0"'
                                  ' loads the KERAS "mod" model from "./<WorkDir>/mod/OSMClassifier_1000.krs"'
                                  ' and generates model statistics and graphics without further training'))
        # --train
        parser.add_argument("--train", dest="train", default=-1, type=int,
                            help=("The number of training epochs (iterations). Ignored if not valid for model."))
        # --check
        parser.add_argument("--check", dest="checkPoint", default=-1, type=int,
                            help=('Number of iterations the training model is saved. Statistics are generated'
                                  ' at each checkpoint. Must be used with --train e.g. "--train 2000 --check 500".'))
        # --version
        parser.add_argument("--version", action="version", version=__version__)

        ExecEnv.args = parser.parse_args()


        # List the available models and exit.
        if ExecEnv.args.modelDescriptions:
            ExecEnv.log.info(ExecEnv.list_available_models())
            sys.exit()

        # Check for a valid model.
        if ExecEnv.selected_model() is None:
            ExecEnv.log.error('Unknown model prefix %s', ExecEnv.args.classifyType)
            ExecEnv.log.info(ExecEnv.list_available_models())
            sys.exit()

        # Check that the work directory exists and terminate if not.
        if not os.path.isdir(ExecEnv.args.workDirectory):
            ExecEnv.log.error('The OSM_QSAR work directory: "%s" does not exist.', ExecEnv.args.workDirectory)
            ExecEnv.log.error("Create or Rename the work directory.")
            ExecEnv.log.error('Please examine the --dir" and "--help" flags.')
            sys.exit()

        # Check that the specified model postfix exists
        if ExecEnv.selected_model() is None:
            ExecEnv.log.warning("No classification model found for prefix: %s", ExecEnv.args.classifyType)
            ExecEnv.log.warning('Use the "--model" flag to see the available classification models.')
            sys.exit()

        # Check to see if the postfix directory and subdirectories exist and create if necessary.
        postfix_directory = os.path.join(ExecEnv.args.workDirectory, ExecEnv.args.classifyType)
        test_directory = os.path.join(postfix_directory, "test")
        train_directory = os.path.join(postfix_directory, "train")
        # Append the postfix directory to the environment file names.
        ExecEnv.args.postfixDirectory = postfix_directory
        ExecEnv.args.testDirectory = test_directory
        ExecEnv.args.trainDirectory = train_directory

        try:
            if not os.path.isdir(postfix_directory):
                ExecEnv.log.info('The model <postfix> directory: "%s" does not exist. Creating it.', postfix_directory)
                os.makedirs(postfix_directory)
            if not os.path.isdir(test_directory):
                ExecEnv.log.info('The model <postfix> directory: "%s" does not exist. Creating it.', test_directory)
                os.makedirs(test_directory)
            if not os.path.isdir(train_directory):
                ExecEnv.log.info('The model <postfix> directory: "%s" does not exist. Creating it.', train_directory)
                os.makedirs(train_directory)
        except OSError:
            ExecEnv.log.error("Could not create directory")
            ExecEnv.log.error("Check the work directory: %s permissions.", ExecEnv.args.workDirectory)
            sys.exit()

        if ExecEnv.args.cleanFlag:
        # clean the postfix subdirectories if the "--clean" flag is specified..
            ExecEnv.log.info('"--clean" specified, deleting all files in directory: "%s"', test_directory)
            ExecEnv.log.info('"--clean" specified, deleting all files in directory: "%s"', train_directory)
            try:
                for file_name in os.listdir(test_directory):
                    file_path = os.path.join(test_directory, file_name)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                for file_name in os.listdir(train_directory):
                    file_path = os.path.join(train_directory, file_name)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
            except OSError:
                ExecEnv.log.error('Specified the "--clean" flag. Could not delete file(s)')
                ExecEnv.log.error("Check <postfix> subdirectories and file permissions.")
                sys.exit()


        ExecEnv.args.dataFilename = os.path.join(ExecEnv.args.workDirectory, ExecEnv.args.dataFilename)

        if ExecEnv.args.loadFilename != "noload":
            ExecEnv.args.loadFilename = os.path.join(postfix_directory,ExecEnv.args.loadFilename)

        ExecEnv.args.saveFilename = os.path.join(postfix_directory,ExecEnv.args.saveFilename)

        ExecEnv.args.logFilename = os.path.join(ExecEnv.args.workDirectory,ExecEnv.args.logFilename)

        if ExecEnv.args.newLogFilename != "nonewlog" and ExecEnv.args.newLogFilename is not None:
            ExecEnv.args.newLogFilename = os.path.join(ExecEnv.args.workDirectory,ExecEnv.args.newLogFilename)
            log_append = False
            self.setup_file_logging(ExecEnv.args.newLogFilename, log_append, file_log_format)

        elif ExecEnv.args.newLogFilename != "nonewlog":  # No filename supplied (optional arg).
            ExecEnv.args.newLogFilename = os.path.join(ExecEnv.args.workDirectory,"OSM_QSAR.log")
            log_append = False
            self.setup_file_logging(ExecEnv.args.newLogFilename, log_append, file_log_format)

        else:
            log_append = True
            self.setup_file_logging(ExecEnv.args.logFilename, log_append, file_log_format)

        # Check that the data file exists and terminate if not.
        if not os.path.exists(ExecEnv.args.dataFilename):
            ExecEnv.log.error('The OSM_QSAR data file: "%s" does not exist.', ExecEnv.args.dataFilename)
            ExecEnv.log.error('Please examine the "--dir", "--data" and "--help" flags.')
            sys.exit()

        # Set up the classification variables.
        ExecEnv.setup_variables()

        cmd_line = ""
        for argStr in sys.argv:
            cmd_line += argStr + " "

        ExecEnv.cmdLine = cmd_line

        # Update the args in the classifier singletons.
        for instance in ExecEnv.modelInstances:
            instance.model_update_environment(ExecEnv.args)

    @staticmethod
    def list_available_models():

        model_str = "A list of available classification models:\n\n"

        for model in ExecEnv.modelInstances:

            model_name = model.model_name() + "\n"
            model_str += model_name
            model_str += "=" * len(model_name) + "\n"
            model_postfix = "--classify "+ model.model_postfix() + "\n"
            model_str += model_postfix
            model_str += "-" * len(model_postfix) + "\n"
            model_str += model.model_description() + "\n\n"

        return model_str

    @staticmethod
    def selected_model():
        model = None
        for instance in ExecEnv.modelInstances:
            if instance.model_postfix() == ExecEnv.args.classifyType:
                model = instance
                break
        return model

    def setup_logging(self, log_format):
        """Set up Python logging"""

        logger = logging.getLogger("OSMLogger")
        logger.setLevel(logging.INFO)  # Default output level.

        # Create a console log

        console_log = logging.StreamHandler()
        console_log.setLevel(logging.DEBUG)  # Output debug to screen
        console_log.setFormatter(log_format)

        logger.addHandler(console_log)

        return logger

    def setup_file_logging(self, log_filename, append, log_format):
        """Set up Python logging to log file"""

        # Create a file log.

        if append:
            file_log = logging.FileHandler(log_filename, mode='a')
        else:
            file_log = logging.FileHandler(log_filename, mode='w')

        file_log.setLevel(logging.INFO)  # Info level and above to file.
        file_log.setFormatter(log_format)

        ExecEnv.log.addHandler(file_log)
        if not append:
            ExecEnv.log.info("Flushed logfile: %s", log_filename)
        ExecEnv.log.info("Logging to file: %s", log_filename)

    @staticmethod
    def setup_variables():

        if ExecEnv.args.indepList == "default": return   # no var list required.

        var_list = [x.strip() for x in ExecEnv.args.indepList.split(',')]
        if len(var_list) == 0:
            ExecEnv.log.error('The "--indep" argument: %s  is incorrectly formatted.', ExecEnv.args.indepList)
            ExecEnv.log.error('Please examine the "--indep" and "--help" flags.')
            sys.exit()

        ExecEnv.args.indepList = var_list

