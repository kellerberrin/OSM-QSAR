OSM_QSAR
========

Open Source Malaria Ligand Classification Toolkit
-------------------------------------------------

A modular toolkit that simplifies the development of user models for 
predicting ligand potency. 

This software currently resides in the GitHub repository:
https://github.com/kellerberrin/OSM-QSAR

OSM_QSAR software is designed to be a software toolkit to simplify the development of 
ligand molecular screening and classification using Machine Learning techniques. 
User models are designed to "plug-in" to OSM_QSAR with minimal coding. 
A neural network classifier can be specified and tested with 20 lines of python code. 
All user specified models automatically have access to statistical and graphical analysis.
See "OSMKeras.py" for an example neural network classifiers using the KERAS
package. The source file "OSMTemplate.py" provides empty templates for researchers to
specify new classification or regression models.

The OSM_QSAR software was developed using Python 2.7. It has also been tested and is  
compatible with a Python 3.5 environment.

For easy installation and package mangement I strongly recommend (mandatory for Windows users) 
using OSM_QSAR with Anaconda python (installation below). The Anaconda python distribution 
can be used on Windows, Linux or Mac. It is designed for data scientists and comes pre-loaded 
with many scientific software tools  and packages. It also has powerful package management features. 
The use of "virtual" environments (see below) largely avoid package version problems that 
commonly occur between collaborating researchers on different computers. 
It also has the excellent "conda" package management tool which greatly simplifies the installation of large
and complex packages such as TENSORFLOW.  
        
Hardware requirements:

A 64 bit Intel/AMD PC or laptop with at least 4GB memory and 10GB free disk space.  
This installation procedure does not cover the hardware/software installation of CUDA GPUs 
to accelerate computation.    
    
The fast install procedure.
===========================

Install on Windows
------------------

1. Install Anaconda (Feb 2017 - python 3.6 64 bit) from https://www.continuum.io/downloads

2. Create the Anaconda virtual environment. This step can take some time (coffee)
as conda downloads and configures the necessary packages.

Create a cmd window (you will need administrator privileges if you installed
Anaconda for "all users") and type:

>conda create -n yourenvname python=3.5 anaconda

Choose python 3.5 as your virtual python version as some packages such as TENSORFLOW
and KERAS are not yet compatible (Feb 2017) with python 3.6. The virtual environment name 
<yourenvname> can be any name. Note that you can also install a python 2.7 virtual
environment if you prefer (install instructions are the same).

3. Activate the virtual environment and install TENSORFLOW, KERAS and RDKIT.

>activate yourenvname

(a) Install TENSORFLOW (non GPU version)

(yourenvname) >conda install -c conda-forge tensorflow

(b) Install RDKIT

(yourenvname) >conda install -c rdkit rdkit

(c) Install KERAS


(yourenvname) >conda install -c conda-forge keras=2.0.2

4. Download the OSM_QSAR files from GitHub (https://github.com/kellerberrin/OSM-QSAR) 
and place the files in a directory of you choice,
cd to your directory. Create a subdirectory "./Work" (case important) and place the 
data file "OSMData,csv" in this directory. The "./Work" directory receives log 
files, model files, molecular images, statistics files, etc, etc. It can be changed 
using the "--dir" flag.

Finally execute:

(yourenvname) >python OSM_QSAR.py 

This executes the neural network classifier coded by Vito Spadavecchio (the default model "seq") 
and generates a log file in ".Data/seq/OSM_QSAR.log" 
(model directories are automatically created), statistics files in
".Data/seq/test/OSMStatistics.csv" and ".Data/seq/train/OSMStatistics.csv",
a model file ".Data/seq/OSMClassify.mdl" and various graphics files 
in the ".Data/seq/test/" and ".Data/seq/train/" directories (requires "--extend"). 
These file names and much else can be customized using
optional command line arguments. These are explained in the usual way 
(the python convention is --flag not -flag).

The help flag shows all the classification options (there are many).

(yourenvname) >python OSM_QSAR.py --help 

Check the OSM_QSAR version.

(yourenvname) >python OSM_QSAR.py --version

See which ligand classification models are implemented. It is very easy to implement
your own classification model and a model template file is provided "OSMTemplate.py".
User models are designed to "plug-in" to OSM_QSAR without further coding.

(yourenvname) >python OSM_QSAR.py --model

Classify the OSM ligands using ML models such "mod", "svmr" and "svmc" and generate 
statistical and graphical output for these models.

(yourenvname) >python OSM_QSAR.py --classify "svmc"

That's it! You are now up and running. Be sure to fork OSM_QSAR 
and post any updated code on GitHub.

Exit your virtual environment:

(yourenvname) >deactivate
 

Install on Linux
------------------

The same as the Windows installation except that you need to prepend "source" when
activating or deactivating a virtual environment.

$source activate yourenvname

$source deactivate


Install on Mac
--------------

A Mac installation has not yet been tested (I don't have a Mac). But it is almost 
certainly the same as the Windows/Linux installation. 


Miscellany.
-----------

To install OSMClassify in a python 2.7 environment, create a virtual environment
and proceed to install the various packages as above. In Anaconda you can have an arbitrary
number of virtual environments.

>conda create -n yourenvname python=2.7 anaconda

To remove an Anaconda virtual environment.

>conda remove --all -n yourenvname

To list all your virtual environments. The current active environment is marked
with a star.

>conda info -e

or

>conda info --envs


You will always have 1 virtual environment called root.


Gotchas and Annoyances.
-----------------------

The Google Neural Network classifier TENSORFLOW is currently under intense development 
and the release candidates are not always stable.
The python 3.5 version I downloaded using conda emitted some annoying but harmless error messages.
You could always try downloading Tensorflow using pip or pip3. See the installation information
on the TENSORFLOW website.

(yourenvname) >python -m pip install tensorflow

Or perhaps download the latest version from git and install it (plan to have a free day).
Remember, with virtual environments you cannot break anything.
If there is a problem, simply delete the virtual environment and start again.

Finally, OSM_QSAR is software under development and may (will) contain bugs. 
If you encounter any bugs, see if you can re-produce the problem and then send me
the details in an email to:

james.duncan.mcculloch@gmail.com
 
Or alternatively, register an issue on GitHub.


