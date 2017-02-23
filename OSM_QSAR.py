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
import os
import sys
import argparse
import logging

# Import the various classification models.
from OSMBase import __modelClassRegistry__
from OSMBase import get_model_method, get_model_instance, get_model_instances
from OSMKeras import SequentialModel
from OSMKeras import ModifiedSequential
from OSMTemplate import OSMTemplateModel  # Classifier template.
from OSMProperties import Properties  # Generate ligand molecular properties.

__version__ = "1.0"


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

        classify_help = "Specify classification model(s) using postfix codes. Valid models: "
        classify_help += model_postfix + '(default "seq").'

        # Parse the runtime args

        parser = argparse.ArgumentParser(
            description="OSM_QSAR. Classification of OSM ligands using machine learning techniques.")
        # --dir
        parser.add_argument("--dir", dest="workDirectory", default="./Work/",
                            help=('The work directory where log files, data, statistics, graphics and model files are found.'
                                  ' Use a Linux style directory specification with trailing forward slash "/"'
                                  ' (default "./Work/").'))
        # --data
        parser.add_argument("--data", dest="dataFilename", default="OSMData.csv",
                            help='The input data filename (default "OSMData.csv").')
        # --load
        parser.add_argument("--load", dest="loadFilename", default="noload",
                            help=("Loads the saved model and generates statistics but does no further training."
                                 "Do not specify a file extension. The file extension is inferred from "
                                 "the model postfix."))
        # --retrain
        parser.add_argument("--retrain", dest="retrainFilename", default="noretrain",
                            help=("Loads the saved model, retrains and generates statistics."
                                 "Do not specify a file extension. The file extension is inferred from "
                                 "the model postfix."))
        # --save
        parser.add_argument("--save", dest="saveFilename", default="OSMClassifier",
                            help=('File name to save the model (default "OSMClassifier").'
                                  "Do not specify a file extension. The file extension is inferred from "
                                  "the model postfix."))
        # --stats
        parser.add_argument("--stats", dest="statsFilename", default="OSMStatistics.csv",
                            help=('File to append the model(s) statistics (default "OSMStatistics.csv").'
                                 ' The model code is automatically post-fixed to the filename.'))
        # --newstats
        parser.add_argument("--newstats", dest="newStatsFilename", default="nonewstats", nargs='?',
                            help=('Flush an existing stats file. The model code is automatically post-fixed'
                                 ' to the filename. (file name argument optional, default "OSMStatistics.csv").'))
        # --roc
        parser.add_argument("--roc", dest="rocGraph", action="store_true",
                            help=('Generate a Receiver Operating Characteristic (ROC) graph for specified model(s).'))
         # --log
        parser.add_argument("--log", dest="logFilename", default="OSM_QSAR.log",
                            help='Log file. Appends the log to any existing logs (default "OSM_QSAR.log").')
        # --newlog
        parser.add_argument("--newlog", dest="newLogFilename", default="nonewlog", nargs='?',
                            help='Flush an existing log file (file name argument optional, default "OSM_QSAR.log").')
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
                                  ' EC50 > 1000 nMol. These thresholds are used to generate model ROC and AUC'
                                  ' statistics and (optionally) the ROC graph.'))
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

        # Check that the work directory exists and terminate if not.

        if not os.path.isdir(ExecEnv.args.workDirectory):
            ExecEnv.log.error('The OSM_QSAR work directory: "%s" does not exist.', ExecEnv.args.workDirectory)
            ExecEnv.log.error("Create or Rename the work directory.")
            ExecEnv.log.error('Please examine the --dir" and "--help" flags.')
            ExecEnv.log.fatal("OSM_QSAR cannot continue.")
            sys.exit()

        # Append the work directory to the environment file names.

        ExecEnv.args.dataFilename = ExecEnv.args.workDirectory + ExecEnv.args.dataFilename
        if ExecEnv.args.loadFilename != "noload":
            ExecEnv.args.loadFilename = ExecEnv.args.workDirectory + ExecEnv.args.loadFilename

        if ExecEnv.args.retrainFilename != "noretrain":
            ExecEnv.args.retrainFilename = ExecEnv.args.workDirectory + ExecEnv.args.retrainFilename

        ExecEnv.args.saveFilename = ExecEnv.args.workDirectory + ExecEnv.args.saveFilename
        ExecEnv.args.statsFilename = ExecEnv.args.workDirectory + ExecEnv.args.statsFilename

        if ExecEnv.args.newStatsFilename != "nonewstats":
            ExecEnv.args.newStatsFilename = ExecEnv.args.workDirectory + ExecEnv.args.newStatsFilename

        ExecEnv.args.logFilename = ExecEnv.args.workDirectory + ExecEnv.args.logFilename

        if ExecEnv.args.newLogFilename != "nonewlog" and ExecEnv.args.newLogFilename is not None:
            ExecEnv.args.newLogFilename = ExecEnv.args.workDirectory + ExecEnv.args.newLogFilename
            log_append = False
            self.setup_file_logging(ExecEnv.args.newLogFilename, log_append, file_log_format)

        elif ExecEnv.args.newLogFilename is not None:  # No filename supplied (optional arg).
            ExecEnv.args.newLogFilename = ExecEnv.args.workDirectory + "OSM_QSAR.log"
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

        # Create a command line string

        for argStr in sys.argv:
            ExecEnv.cmdLine += argStr + " "

        # Update the args in the classifier singletons.

        for instance in ExecEnv.modelInstances:
            instance.update_args(ExecEnv.args)


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


    @staticmethod
    def list_available_models():

        model_str = "A list of available classification models:\n\n"

        for model in ExecEnv.modelInstances:

            model_name = model.model_name() + "\n"
            model_str += model_name
            model_str += "=" * len(model_name) + "\n"
            model_postfix = "Postfix (classify):"+ model.model_postfix() + "\n"
            model_str += model_postfix
            model_str += "-" * len(model_postfix) + "\n"
            model_str += model.model_description() + "\n\n"

        return model_str


# ===================================================================================================
# This is a high level object that implements the various classification models.
# ====================================================================================================


def main():

    try:

        ExecEnv()  # Setup the runtime environment.

        # List the available models and exit.

        if ExecEnv.args.modelDescriptions:
            ExecEnv.log.info(ExecEnv.list_available_models())
            sys.exit()

        # Perform classification(s) and analytics.

        ExecEnv.log.info("############ OSM_QSAR %s Start Classification ###########", __version__)
        ExecEnv.log.info("Command Line: %s", ExecEnv.cmdLine)

        prop_obj = Properties(ExecEnv.args, ExecEnv.log)  # Use the ligand SMILEs to generate molecular properties


        classify = False
        for instance in ExecEnv.modelInstances:
            if instance.model_postfix() == ExecEnv.args.classifyType:
                instance.classify(prop_obj.train, prop_obj.test)
                classify =True

        if not classify:
            ExecEnv.log.warning("No classification model found for prefix: %s", ExecEnv.args.classifyType)
            ExecEnv.log.warning('Use the "--model" flag to see the available classification models.')


        ExecEnv.log.info("Command Line: %s", ExecEnv.cmdLine)
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
