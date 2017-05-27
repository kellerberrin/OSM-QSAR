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

"""
Script that trains graph-conv models on Tox21 dataset.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import time
import numpy as np
#np.random.seed(123)


import tensorflow as tf
#tf.set_random_seed(123)
import deepchem as dc


def load_tox21(featurizer='ECFP', split='index'):
  """Load Tox21 datasets. Does not do train/test split"""
  # Featurize Tox21 dataset
  print("About to featurize Tox21 dataset.")
  current_dir = os.path.dirname(os.path.realpath(__file__))
  dataset_file = os.path.join(
      current_dir, "./Work/OSMFinalConvTest.csv")


  tox21_tasks = ['ION_ACTIVITY', 'EC50_200', 'EC50_500', 'EC50_1000']


  if featurizer == 'ECFP':
    featurizer_func = dc.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer_func = dc.feat.ConvMolFeaturizer()
  loader = dc.data.CSVLoader(
      tasks=tox21_tasks, smiles_field="SMILE", id_field="ID", featurizer=featurizer_func)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  # Initialize transformers
  transformers = [
      dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)]

  print("About to transform data")
  for transformer in transformers:
      dataset = transformer.transform(dataset)

  splitters = {'index': dc.splits.IndexSplitter(),
               'random': dc.splits.RandomSplitter(),
               'scaffold': dc.splits.ScaffoldSplitter(),
               'butina': dc.splits.ButinaSplitter()}

  splitter = dc.splits.IndexSplitter()
  splitter = dc.splits.IndiceSplitter(True, range(703,1103), [])

  train, valid, test = splitter.train_valid_test_split(dataset)

  splitter = dc.splits.RandomSplitter()

  train, not_valid, test = splitter.train_valid_test_split(dataset=train, frac_test=0.2,
                                                           frac_train=0.8, frac_valid=0.0 )

  print("len(train)", len(train), "len(test)", len(test), "len(valid)", len(valid))

  return tox21_tasks, (train, valid, test), transformers



# Load Tox21 dataset
tox21_tasks, tox21_datasets, transformers = load_tox21(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = tox21_datasets

print("test_dataset")
print(test_dataset)


# Fit models
metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, np.mean, mode="classification")

# Number of features on conv-mols
n_feat = 75
# Batch size of models
batch_size = 50
graph_model = dc.nn.SequentialGraph(n_feat)
graph_model.add(dc.nn.GraphConv(128, n_feat, activation='relu'))
graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
graph_model.add(dc.nn.GraphPool())
graph_model.add(dc.nn.GraphConv(128, 128, activation='relu'))
graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
graph_model.add(dc.nn.GraphPool())
# Gather Projection
graph_model.add(dc.nn.Dense(128, 128, activation='relu'))
graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
graph_model.add(dc.nn.GraphGather(batch_size, activation="tanh"))

model = dc.models.MultitaskGraphClassifier(
    graph_model,
    len(tox21_tasks),
    n_feat,
    batch_size=batch_size,
    learning_rate=1e-3,
    learning_rate_decay_time=1000,
    optimizer_type="adam",
    beta1=.9,
    beta2=.999)

# Fit trained model
model.fit(train_dataset, nb_epoch=20)

print("Evaluating model")
train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)

prediction = model.predict_proba(valid_dataset)
print("Prediction")

ids_list = list(valid_dataset.ids)
y_list = list(valid_dataset.y)

prob_list = []
for idx, pred in enumerate(prediction):
    prob_list.append([ids_list[idx], pred[0][0], y_list[idx][0]])

RESULTDATAFILE = "./Work/Run10.csv"

# Open the data file
outFile = open(RESULTDATAFILE, "w")

if outFile.closed:
    print("Unable to open reult output file")
    sys.exit()    # Terminate with extreme prejudice


for Line in prob_list:
    textLine = "{} , {}\n".format(Line[0], Line[1])
    outFile.write(textLine)

# Close the data file
outFile.close()

#print(len(prob_list), prob_list)