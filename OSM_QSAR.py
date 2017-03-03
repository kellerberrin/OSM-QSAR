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
import math
import time

# Import the various classification models.
from OSMBase import get_model_instances
from OSMProperties import Properties  # Generate ligand molecular properties.
from OSMKeras import SequentialModel, ModifiedSequential
from OSMTemplate import OSMTemplateModel
from OSMSKLearn import OSMSKLearnSVMR


__version__ = "0.1"


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
                                  ' of this directory. Use a Linux style directory specification with trailing forward'
                                  ' slash "/" (default "./Work/").'
                                  " Important - to run OSM_QSAR this directory must exist, it will not be created."
                                  ' Model specific files (models, statistics and graphics) are in the '
                                  'subdirectories "/<WorkDir>/postfix/".'))

        # --data
        parser.add_argument("--data", dest="dataFilename", default="OSMData.csv",
                            help=('The input data filename (default "OSMData.csv").'
                                 " Important - to run OSM_QSAR this file must exist in the Work directory."
                                 ' For example, if this flag is not specified then OSM_QSAR attempts to read the'
                                 ' data file at "/<WorkDir>/OSMData.csv".'
                                 " See the additional OSM_QSAR documentation for the format of this file."))

        # --load
        parser.add_argument("--load", dest="loadFilename", default="noload",
                            help=("Loads the saved model and generates statistics and graphics but does no "
                                  " further training."
                                  " This file is always located in the model <postfix> subdirectory of the Work"
                                  ' directory. For example, specifying "mymodel.mdl"'
                                  ' loads "./<WorkDir>/<postfix>/mymodel.mdl".'))
        # --retrain
        parser.add_argument("--retrain", dest="retrainFilename", default="noretrain",
                            help=("Loads the saved model, retrains and generates statistics and graphics."
                                  " This file is always located in the model <postfix> subdirectory of the Work"
                                  ' directory. For example, specifying "mymodel.mdl"'
                                  ' loads "./<WorkDir>/<postfix>/mymodel.mdl".'))
        # --save
        parser.add_argument("--save", dest="saveFilename", default="OSMClassifier.mdl",
                            help=('File name to save the model (default "OSMClassifier.mdl").'
                                  ' The model file is always saved to the model postfix subdirectory.'
                                  ' For example, if this flag is not specified the model is saved to:'
                                  ' "./<WorkDir>/postfix/OSMClassifier.mdl".'
                                  ' The postfix directory is created if it does not exist.'))
        # --stats
        parser.add_argument("--stats", dest="statsFilename", default="OSMStatistics.csv",
                            help=('File to append the test and train model statistics (default "OSMStatistics").'
                                  ' The statistics file is always saved to a subdirectory the model <postfix>'
                                  ' directory. Test data statistics are always appended to the specified statisitcs'
                                  ' file in the "./<WorkDir>/<postfix>/test/". directory.'
                                  ' If the "--extend" flag is specified then training data statistics are appended'
                                  ' to the specified statistics file in the "./<WorkDir>/<postfix>/train" directory.'
                                  ' The postfix directory and the "test" and "train" subdirectories are created'
                                  ' if they do not exist. The statistics file(s) are created if they do not exist.'))
        # --extend
        parser.add_argument("--extend", dest="extendFlag", action="store_true",
                            help=(' The "--extend" flag generates training data statistics and graphics.'
                                  ' Training data statistics are appended to the statistics file in the'
                                  ' "./<WorkDir>/<postfix>/train" directory.'
                                  ' In addition, training data graphics files are deposited to the same directory.'
                                  ' The directory is created is it does not exist. The statistics file is created'
                                  ' if it does not exist.'
                                  ' Warning - the "--extend" flag may substantially increase OSM_QSAR runtime.'))

        # --clean
        parser.add_argument("--clean", dest="cleanFlag", action="store_true",
                            help=('Deletes all files in the "test" and "train" subdirectories of the model '
                                  ' <postfix> directory before OSM_QSAR executes.'
                                  ' The model files in the <postfix> directory are not deleted.'))
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
                            help=("Lists all defined classification models and exits."))
        # --active
        parser.add_argument("--classify", dest="classifyType", default="seq", help=classify_help)
        # --active
        parser.add_argument("--active", dest="activeNmols", default="active : 200.0",
                            help=('Define the ligand Active/Inactive EC50 classification thresholds in nMols'
                                  ' (default "active : 200.0"). Quotes are always required. Any number of thresholds'
                                  ' and classifications can be specified. For example: "active : 200, partial : 350,'
                                  ' ainactive : 600, binactive : 1000" defines 5 classifications and thresholds.'
                                  ' "inactive" is always implied and in the example will be any ligand with an'
                                  ' EC50 > 1000 nMol. These thresholds are used to specify the potency classes'
                                  ' used by classification models.'))
        # --epoch
        parser.add_argument("--epoch", dest="epoch", default=-1, type=int,
                            help='The number of training epochs (iterations). Ignored if not valid for model.')
        # --check
        parser.add_argument("--check", dest="checkPoint", default=-1, type=int,
                            help=('Number of iterations the training model is saved. Statistics are generated'
                                  ' at each checkpoint. Must be used with --epoch e.g. "--epoch 2000 --check 500".'))
        # --version
        parser.add_argument("--version", action="version", version=__version__)

        ExecEnv.args = parser.parse_args()


        # List the available models and exit.

        if ExecEnv.args.modelDescriptions:
            ExecEnv.log.info(ExecEnv.list_available_models())
            sys.exit()

        # Check that the work directory exists and terminate if not.

        if not os.path.isdir(ExecEnv.args.workDirectory):
            ExecEnv.log.error('The OSM_QSAR work directory: "%s" does not exist.', ExecEnv.args.workDirectory)
            ExecEnv.log.error("Create or Rename the work directory.")
            ExecEnv.log.error('Please examine the --dir" and "--help" flags.')
            ExecEnv.log.fatal("OSM_QSAR cannot continue.")
            sys.exit()

        # Check that the specified model postfix exists

        classify = False
        for instance in ExecEnv.modelInstances:
            if instance.model_postfix() == ExecEnv.args.classifyType:
                classify =True

        if not classify:
            ExecEnv.log.warning("No classification model found for prefix: %s", ExecEnv.args.classifyType)
            ExecEnv.log.warning('Use the "--model" flag to see the available classification models.')
            ExecEnv.log.fatal("OSM_QSAR cannot continue.")
            sys.exit()

        # Check to see if the postfix directory and subdirectories exist and create if necessary.

        postfix_directory = os.path.join(ExecEnv.args.workDirectory, ExecEnv.args.classifyType)
        test_directory = os.path.join(postfix_directory, "test")
        train_directory = os.path.join(postfix_directory, "train")
        # Append the postfix directory to the environment file names.
        ExecEnv.args.graphicsDirectory = postfix_directory # ********remove this line.
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
            ExecEnv.log.fatal("OSM_QSAR cannot continue.")
            sys.exit()

        # clean the postfix subdirectories if the "--clean" flag is specified..

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
            ExecEnv.log.fatal("OSM_QSAR cannot continue.")
            sys.exit()


        ExecEnv.args.dataFilename = os.path.join(ExecEnv.args.workDirectory, ExecEnv.args.dataFilename)

        if ExecEnv.args.loadFilename != "noload":
            ExecEnv.args.loadFilename = os.path.join(postfix_directory,ExecEnv.args.loadFilename)

        if ExecEnv.args.retrainFilename != "noretrain":
            ExecEnv.args.retrainFilename = os.path.join(postfix_directory,ExecEnv.args.retrainFilename)

        ExecEnv.args.saveFilename = os.path.join(postfix_directory,ExecEnv.args.saveFilename)

        ExecEnv.args.logFilename = os.path.join(ExecEnv.args.workDirectory,ExecEnv.args.logFilename)

        if ExecEnv.args.newLogFilename != "nonewlog" and ExecEnv.args.newLogFilename is not None:
            ExecEnv.args.newLogFilename = os.path.join(ExecEnv.args.workDirectory,ExecEnv.args.newLogFilename)
            log_append = False
            self.setup_file_logging(ExecEnv.args.newLogFilename, log_append, file_log_format)

        elif ExecEnv.args.newLogFilename is not None:  # No filename supplied (optional arg).
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
            ExecEnv.log.fatal("OSM_QSAR cannot continue.")
            sys.exit()

        # Set up the classification arguments.

        ExecEnv.args.activeNmols = self.classification_array(ExecEnv.args.activeNmols)

        # Create some additional entries in the argument namespace for cmd line and cpu time variables..

        cmd_line = ""
        for argStr in sys.argv:
            cmd_line += argStr + " "

        ExecEnv.cmdLine = cmd_line

        # Update the args in the classifier singletons.

        for instance in ExecEnv.modelInstances:
            instance.model_update_args(ExecEnv.args)



    @staticmethod
    def list_available_models():

        model_str = "A list of available classification models:\n\n"

        for model in ExecEnv.modelInstances:

            model_name = model.model_name() + "\n"
            model_str += model_name
            model_str += "=" * len(model_name) + "\n"
            model_postfix = "Postfix (--classify):"+ model.model_postfix() + "\n"
            model_str += model_postfix
            model_str += "-" * len(model_postfix) + "\n"
            model_str += model.model_description() + "\n\n"

        return model_str


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
        if append:
            ExecEnv.log.info("Flushed logfile: %s", log_filename)
        ExecEnv.log.info("Logging to file: %s", log_filename)

    def classification_array(self, arg_string):
        """Return as a sorted array of tuples"""
        sorted_array = []
        try:
            classify_array = [x.strip() for x in arg_string.split(',')]
            for element in classify_array:
                atoms = [x.strip() for x in element.split(':')]
                pEC50_uMols = math.log10(float(atoms[1]) / 1000.0) # Convert from nMols to log10 uMols
                sorted_array.append((pEC50_uMols,atoms[0]))

            sorted(sorted_array, key=lambda x: x[0])  # Ensure ordering.

            if len(sorted_array) == 0:
                raise ValueError

        except:
            ExecEnv.log.error('The "--active" argument: %s  is incorrectly formatted.', arg_string)
            ExecEnv.log.error('Please examine the "--active" and "--help" flags.')
            ExecEnv.log.fatal("OSM_QSAR Cannot Continue.")
            sys.exit()

        return sorted_array


# ===================================================================================================
# The program mainline.
# ====================================================================================================


def main():

    try:


        ExecEnv()  # Setup the runtime environment.


        ExecEnv.log.info("############ OSM_QSAR %s Start Classification ###########", __version__)
        ExecEnv.log.info("Command Line: %s", ExecEnv.cmdLine)

        prop_obj = Properties(ExecEnv.args, ExecEnv.log)  # Use the ligand SMILEs to generate molecular properties

        for instance in ExecEnv.modelInstances:
            if instance.model_postfix() == ExecEnv.args.classifyType:
                instance.classify(prop_obj.train, prop_obj.test)

        ExecEnv.log.info("Command Line: %s", ExecEnv.cmdLine)
        ExecEnv.log.info("Elapsed seconds CPU time %f (all processors, may exceed clock time, assumes no GPU)."
                         , time.clock())
        ExecEnv.log.info("############ OSM_QSAR %s End Classification ###########", __version__)

    except KeyboardInterrupt:
        ExecEnv.log.warning("\n\n")
        ExecEnv.log.warning("Control-C pressed. Program terminates. Open files may be in an unsafe state.")

    except IOError:
        ExecEnv.log.fatal(
            'File error. Check filenames. Check the default work directory "--dir" with "OSM_QSAR.py --help".')

    except SystemExit:
        clean_up = None  # Placeholder for any cleanup code.

if __name__ == '__main__':
    main()
