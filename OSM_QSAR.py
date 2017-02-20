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
import csv
import math
import operator
import numpy
import argparse
import logging


# Import the various classification models.
from OSMKeras import SequentialModel
from OSMKeras import ModifiedSequential
from OSMTemplate import OSMNewModel   # Classifier template.
from OSMProperties import Properties  # Generate ligand molecular properties.


__version__ = 1.0

# ===================================================================================================
# A utility class to parse the program runtime arguments
# and setup a logger to receive classification output.
# It is only necessary to specify the first 3 characters
# of a classification model. For example - "seq" and "sequential" 
# are equivalent.

# Note that when specifying a load or save filename it is not necessary
# to specifiy a file extension as this is added by different models.
# For example, if the "seq" model is selected with a save file "--save_model run10", then the
# Sequential model is saved to "run10.seq".

# Current implemented models are:
# "seq" ("Sequential") the sequential model developed by Vito Spadavecchio
# "mod" ("ModifiedSequential") the sequential model modified to reduce (but not eliminate) overfitting.
# "new" ("OSMNewModel") an unimplemented template model provided for the convenience of model developers.
# "all" runs all models (except OSMNewModel) in the order listed above.
# ===================================================================================================


class ExecEnv(object): # Python 2.7 new style objects. Obsolete in Python 3.
    """Utility class to setup the runtime environment and logging"""

# Static class variables.

    args = None   
    log = None
    cmdLine = ""
    logFormat = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


    def __init__(self):
        """Parse runtime arguments on object creation"""

# Start a console logger to complain about bad args.

        ExecEnv.log = self.setuplogging()

# Parse the runtime args

        parser = argparse.ArgumentParser(description="OSM_QSAR. Classification of OSM ligands using machine learning techniques.")
#--dir
        parser.add_argument("--dir", dest="workDirectory", default="./Work/",
            help=('The work directory where log files, data, statistcs and model files are found.'
                  ' Use a Linux style directory specification with trailing forward slash "/" (default "./Work/").'))
#--data
        parser.add_argument("--data", dest="dataFilename", default="OSMData.csv",
            help='The input data filename (default "OSMData.csv").')
#--load
        parser.add_argument("--load", dest="loadFilename", default="noload",
            help="Loads the saved model and generates statistics but does no further training. Do not append file extension.")
#--retrain
        parser.add_argument("--retrain", dest="retrainFilename", default="noretrain",
            help="Loads the saved model, retrains and generates statistics. Do not append file extension.")
#--save
        parser.add_argument("--save", dest="saveFilename", default="OSMClassifier",
            help='File name to save the model. Do not append file extension. (default "OSMClassifier")') 
#--stats
        parser.add_argument("--stats", dest="statsFilename", default="OSMStatistics.csv",
            help='File to append the model(s) statistics (default "OSMStatistics.csv"). The statistics are saved in CSV format.') 
#--newstats
        parser.add_argument("--newstats", dest="newStatsFilename", default="nonewstats", nargs='?',
            help='Flush an existing stats file (file name argument optional, default "OSMStatistics.csv").')
#--roc
        parser.add_argument("--roc", dest="rocGraph", action="store_true",
            help=('Generate a Receiver Operating Characteristic (ROC) graph for specified model(s).'
                  ' May not work on non-linux platforms.'))
#--log
        parser.add_argument("--log", dest="logFilename", default="OSM_QSAR.log",
            help='Log file. Appends the log to any existing logs (default "OSM_QSAR.log").')
#--newlog
        parser.add_argument("--newlog", dest="newLogFilename", default="nonewlog", nargs='?',
            help='Flush an existing log file (file name argument optional, default "OSM_QSAR.log").')
#--model
        parser.add_argument("--model", dest="modelType", default="seq",
            help=('Specify classification model(s). Valid models: "seq", "mod", "new" (default "seq").'
                  ' Multiple models can be specified as: "mod, seq". In this case quotes are required.'))
#--active
        parser.add_argument("--active", dest="activeNmols", default="active : 200.0",
            help = ('Define the ligand Active/Inactive EC50 classification thresholds in nMols (default "active : 200.0").'
                    ' Quotes are always required. Any number of thresholds and classifications can be specified.'
                    ' For example: "active : 200, partial : 350, ainactive : 600, binactive : 1000" defines 5 classifications and thresholds.'
                    ' "inactive" is always implied and in the example will be any ligand with an EC50 > 1000 nMol.' 
                    ' These thresholds are used to generate model ROC and AUC statisitics and (optionally) the ROC graph.'))
#--epoch
        parser.add_argument("--epoch", dest="epoch", default=-1, type=int,
            help='The number of training epochs (iterations). Only valid for "seq" and "mod".')
#--check
        parser.add_argument("--check", dest="checkPoint", default=-1, type=int,
            help=('Number of iterations the training model is saved. Statistics are generated at each checkpoint.'
                  ' Must be used with --epoch e.g. "--epoch 2000 --check 500".'))  
#--version
        parser.add_argument("--version", action="version", version=__version__)

        ExecEnv.args = parser.parse_args()

# Check that the work directory exists and terminate if not.

        if not os.path.isdir(ExecEnv.args.workDirectory):
            ExecEnv.log.error('The OSM_QSAR work directory: "%s" does not exist.', ExecEnv.args.workDirectory)
            ExecEnv.log.error("Create or Rename the work directory.")
            ExecEnv.log.error('Execute "OSM-QSAR.py --help" and examine the "--dir" flag.')
            ExecEnv.log.fatal("OSM_QSAR cannot continue.")
            sys.exit()
                    
# Append the work directory to the environment file names.

        ExecEnv.args.dataFilename = ExecEnv.args.workDirectory + ExecEnv.args.dataFilename
        if ExecEnv.args.loadFilename != "noload":
            ExecEnv.args.loadFilename = ExecEnv.args.workDirectory + ExecEnv.args.loadFilename

        if ExecEnv.args.retrainFilename != "noretrain":             
            ExecEnv.args.retrainFilename = ExecEnv.args.workDirectory

        ExecEnv.args.saveFilename = ExecEnv.args.workDirectory + ExecEnv.args.saveFilename
        ExecEnv.args.statsFilename = ExecEnv.args.workDirectory + ExecEnv.args.statsFilename

        if ExecEnv.args.newStatsFilename != "nonewstats":        
            ExecEnv.args.newStatsFilename = ExecEnv.args.workDirectory + ExecEnv.args.newStatsFilename

        ExecEnv.args.logFilename = ExecEnv.args.workDirectory + ExecEnv.args.logFilename

        if ExecEnv.args.newLogFilename != "nonewlog" and ExecEnv.args.newLogFilename != None:
            ExecEnv.args.newLogFilename = ExecEnv.args.workDirectory + ExecEnv.args.newLogFilename
            logAppend = False
            self.setupfilelogging(ExecEnv.args.newLogFilename, logAppend)

        elif ExecEnv.args.newLogFilename == None:  # No filename supplied (optional arg).          
            ExecEnv.args.newLogFilename = ExecEnv.args.workDirectory + "OSM_QSAR.log"
            logAppend = False
            self.setupfilelogging(ExecEnv.args.newLogFilename, logAppend)

        else:
            logAppend = True
            self.setupfilelogging(ExecEnv.args.logFilename, logAppend)
                    

# Check that the data file exists and terminate if not.

        if not os.path.exists(ExecEnv.args.dataFilename):
            ExecEnv.log.error('The OSM_QSAR data file: "%s" does not exist.', ExecEnv.args.dataFilename)
            ExecEnv.log.error('Please examine the "--dir" and "--data" flags. Execute "OSM_QSAR.py --help".')
            ExecEnv.log.fatal("OSM_QSAR cannot continue.")
            sys.exit()        

# Create a command line string
        
        for argStr in sys.argv:
            ExecEnv.cmdLine += argStr + " "


    def setuplogging(self):
        """Set up Python logging"""

        logger = logging.getLogger("OSMLogger")
        logger.setLevel(logging.INFO) # Default output level.


# Create a console log

        consoleLog = logging.StreamHandler()
        consoleLog.setLevel(logging.DEBUG) # Output debug to screen 
        consoleLog.setFormatter(ExecEnv.logFormat)

        logger.addHandler(consoleLog)

        return logger


    def setupfilelogging(self, logFileName, append):
        """Set up Python logging to log file"""

# Create a file log.

        if append:
            fileLog = logging.FileHandler(logFileName, mode='a')        
        else:
            fileLog = logging.FileHandler(logFileName, mode='w')

        fileLog.setLevel(logging.INFO) # Info level and above to file.
        fileLog.setFormatter(ExecEnv.logFormat)

        ExecEnv.log.addHandler(fileLog)

        ExecEnv.log.info("Logging to file: %s", logFileName)



# ===================================================================================================
# This is a high level object that implements the various classification models.
# Current implemented models are:
# "seq" ("Sequential") the sequential model developed by Vito Spadavecchio
# "mod" ("ModifiedSequential") the sequential model modified to reduce (but not eliminate) overfitting.
# "new" ("NewModel") an unimplemented stub model for the convenience of other model developers.

# "all" ("all_models") runs all models in the order listed above.

# The source file for "seq" and "mod" is "OSMSequential.py".
# The source file for "new" is "OSMNewModel.py".
# These are imported in the header section above.
# ====================================================================================================


class Classification(object):
    """Execute the requested classification model"""


    def __init__(self, train, test):

        modelStr = ExecEnv.args.modelType[:3]
        modelStr.lower() # Case insenstive.

        if modelStr == "seq":

            SequentialModel(train, test, ExecEnv.args, ExecEnv.log)

        elif modelStr == "mod":
        
            ModifiedSequential(train, test, ExecEnv.args, ExecEnv.log)

        elif modelStr == "all":

            SequentialModel(train, test, ExecEnv.args, ExecEnv.log)        
            ModifiedSequential(train, test, ExecEnv.args, ExecEnv.log)

        elif modelStr == "new":
        
            OSMNewModel(train, test, ExecEnv.args, ExecEnv.log)

        else:
        
            ExecEnv.log.warning("****** No Classification Performed *******")
            ExecEnv.log.warning("Invalid classifier model specified: %s. Valid models are: ", ExecEnv.args.modelType)
            ExecEnv.log.warning('"seq" ("Sequential")');
            ExecEnv.log.warning('"mod" ("ModifiedSequential")');
            ExecEnv.log.warning('"new" ("NewModel") - Note that this model is an example template');

        

def main():

    try:

        ExecEnv() # Setup the runtime environment.
        ExecEnv.log.info("############ OSM_QSAR %s Start Classification ###########", __version__)
        ExecEnv.log.info("Command Line: %s", ExecEnv.cmdLine)

        PropObj = Properties(ExecEnv.args, ExecEnv.log) # Use the ligand SMILEs to generate molecular properties 
        Classification(PropObj.train, PropObj.test) # Create the classifier model, train the classifier and report results.

        ExecEnv.log.info("Command Line: %s", ExecEnv.cmdLine)
        ExecEnv.log.info("############ OSM_QSAR %s End Classification ###########", __version__)

    except KeyboardInterrupt:
        ExecEnv.log.warning("\n\n")    
        ExecEnv.log.warning("Control-C pressed. Program terminates. Open files may be in an unsafe state.")

    except IOError:
        ExecEnv.log.fatal('File error. Check filenames. Check the default work directory "--dir" with "OSM_QSAR.py --help".')
            
    except SystemExit:
        cleanup = None    #Placeholder for any cleanup code.

if __name__ == '__main__':
    main()
