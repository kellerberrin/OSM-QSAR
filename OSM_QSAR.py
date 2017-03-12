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


import time

# REad CSVs and Generate molecular properties.
from OSMProperties import OSMGenerateData

# Import the runtime environment object.
from OSMExecEnv import ExecEnv, __version__



# ===================================================================================================
# The program mainline.
# ====================================================================================================

def main():

    try:

        ExecEnv()  # Setup the runtime environment.

        ExecEnv.log.info("############ OSM_QSAR %s Start Classification ###########", __version__)
        ExecEnv.log.info("Command Line: %s", ExecEnv.cmdLine)

        data = OSMGenerateData(ExecEnv.args, ExecEnv.log)  # Create all the variables used in the regressions.

        ExecEnv.selected_model().classify(data) # Do the regression, classification or unsupervised.

        ExecEnv.log.info("Command Line: %s", ExecEnv.cmdLine)
        ExecEnv.log.info("Elapsed seconds CPU time %f (all processors, assumes no GPU).", time.clock())
        ExecEnv.log.info("############ OSM_QSAR %s End Classification ###########", __version__)

    except KeyboardInterrupt:
        ExecEnv.log.warning("\n\n")
        ExecEnv.log.warning("Control-C pressed. Program terminates. Open files may be in an unsafe state.")

    except IOError:
        ExecEnv.log.fatal("File error. Check directories, file names and permissions."
                          ' Check the default work directory "--dir" and --help".')
        ExecEnv.log.fatal("OSM_QSAR exits.")

    except SystemExit:

        ExecEnv.log.info("OSM_QSAR exits.")

    finally:

        clean_up = None  # Placeholder for any cleanup code.

if __name__ == '__main__':
    main()
