"""
Reconciling results from MLJAR and sklearn.
  Random forest classifier
  X,y from scikit-learn make_blobs

Usage
  python ex2-binary-noKFold-blob.py
  or
  >>> exec(open("ex2-binary-noKFold-blob.py").read(), globals())
"""

print(__doc__)

MAX_VERSION_INFO = 3, 0

import sys
if not sys.version_info < MAX_VERSION_INFO:
    exit("Python {}.{}- is required.".format(*MAX_VERSION_INFO))

#################################

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from mljar import Mljar
import pickle
from os import path

#################################
# Data

fnCache = 'ex2-cache.pickle'
if path.exists(fnCache):
  print("load data from cache")
  with open(fnCache, 'rb') as handle:
    X, y = pickle.load(handle)
else:
  print("generate data")
  n_samples = 100
  # Binary classification
  # using make_blobs from sklearn
  X, y = make_blobs(n_samples=n_samples, centers=2, n_features=1, random_state=0)
  with open(fnCache, 'wb') as handle:
    pickle.dump((X, y), handle)

print("X",X.transpose())
print("y",y)

########################
print("MLJAR Random forest classification")
clf_mlj = Mljar(
  project='Recon mljar-sklearn',
  experiment='Ex 3.1', # use ex2 cached data, but with 5-fold cross-validation
  metric='auc',
  algorithms=['rfc'],
  validation_kfolds=5,
  validation_shuffle=False,
  validation_stratify=True,
  validation_train_split=None,
  tuning_mode='Normal',
  create_ensemble=False,
  single_algorithm_time_limit='1'
)

print("fit")
clf_mlj.fit(X,y) #,dataset_title="sklearn make_blobs + k-fold")

# Until https://github.com/mljar/mljar-api-python/issues/2
# gets fixed,
# manually get the result to use in the prediction
# by browsing mljar.com
from mljar.client.result import ResultClient
client = ResultClient(clf_mlj.project.hid)
results = client.get_results(clf_mlj.experiment.hid)
# len(results) # returns 75
# results = client.get_results(None)
# len(results) # also returns 75
# rid = 'RMxezRO0eaGy'
rid = raw_input("Enter result ID from mljar.com (until issue#2 is solved): ")
selected = [x for x in results if x.hid==rid]
if len(selected)!=1:
  raise Exception("len(selected)!=1")

clf_mlj.selected_algorithm = selected[0]


print("predict")
pred_proba_mlj = clf_mlj.predict(X)
pred_proba_mlj = pred_proba_mlj.squeeze().values
print("pred_proba_mlj",pred_proba_mlj) # shows probabilities
print("log loss mlj:",log_loss(y,pred_proba_mlj))
# ('log loss mlj:', 0.08321280267622938)

pred_mlj = [1 if x>0.5 else 0 for x in pred_proba_mlj]
print("mse mlj:",mean_squared_error(y,pred_proba_mlj))
# ('mse mlj:', 0.028762043544997624)

# ('mljar_fit_params', {u'max_features': 0.7, u'min_samples_split': 8, u'criterion': u'entropy', u'min_samples_leaf': 4})
mljar_fit_params = clf_mlj.selected_algorithm.params['model_params']['fit_params']
print("mljar_fit_params", mljar_fit_params)

random_seed = clf_mlj.selected_algorithm.params['random_seed']

########################
print("Random forest with same params")
clf_skl = RandomForestClassifier(
  n_estimators = 5,
  criterion = mljar_fit_params['criterion'],
  max_features = mljar_fit_params['max_features'],
  min_samples_split = mljar_fit_params['min_samples_split'],
  min_samples_leaf = mljar_fit_params['min_samples_leaf'],
  random_state=2016
)

# http://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
clf_skl_sig = CalibratedClassifierCV(clf_skl, cv=5, method='sigmoid')
clf_skl_sig.fit(X, y)

pred_proba_skl = clf_skl_sig.predict_proba(X)
pred_proba_skl = pred_proba_skl[:,pred_proba_skl.shape[1]-1]
print("pred_proba_skl",pred_proba_skl) # shows probabilities between 0 and 1, with plenty of values being 0 or 1
print("log loss skl:",log_loss(y,pred_proba_skl))
# ('log loss skl:', 0.17435998101042061)

pred_skl = clf_skl_sig.predict(X)
print("pred_skl",pred_skl) # shows values = 0 or 1
print("same",(pred_skl==y).all()) # returns False
print("mse skl:",mean_squared_error(y,pred_skl))
# ('mse skl:', 0.050000000000000003)

print(np.matrix([pred_proba_mlj,pred_proba_skl,y]).transpose())
# Lots of differences below. TODO figure out why
# [[ 1.          0.90302887  1.        ]
#  [ 0.04604545  0.14223289  0.        ]
#  [ 0.          0.09639619  0.        ]
#  [ 0.          0.09639619  0.        ]
#  [ 1.          0.90302887  1.        ]
#  [ 1.          0.90302887  1.        ]
#  [ 0.          0.09639619  0.        ]
#  [ 1.          0.90302887  1.        ]
#  ...
