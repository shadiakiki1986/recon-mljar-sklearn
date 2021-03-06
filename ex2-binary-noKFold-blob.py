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
  # experiment='Ex 2.1', # validation_train_split=0.05
  # experiment='Ex 2.2', # manual through website
  # experiment='Ex 2.3', # validation_train_split=0.95
  experiment='Ex 2.4', # cache the data locally at the same time as uploading to be able to replicate results
  metric='auc',
  algorithms=['rfc'],
  validation_kfolds=None,
  validation_shuffle=False,
  validation_stratify=True,
  validation_train_split=0.95,
  tuning_mode='Normal',
  create_ensemble=False,
  single_algorithm_time_limit='1'
)

print("fit")
clf_mlj.fit(X,y) #,dataset_title="sklearn make_blobs")

# from mljar.client.project import ProjectClient
# clf_mlj.project = ProjectClient().create_project_if_not_exists(clf_mlj.project_title, clf_mlj.project_task)

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
rid = raw_input("Enter result ID from mljar.com until issue#2 is solved [RMxezRO0eaGy]: ")
if rid == '': rid = 'RMxezRO0eaGy'

selected = [x for x in results if x.hid==rid]
if len(selected)!=1:
  raise Exception("len(selected)!=1")

clf_mlj.selected_algorithm = selected[0]


print("predict")
pred_proba_mlj = clf_mlj.predict(X)
pred_proba_mlj = pred_proba_mlj.squeeze().values
print("pred_proba_mlj",pred_proba_mlj) # shows probabilities
print("log loss mlj:",log_loss(y,pred_proba_mlj))
# ('log loss mlj:', 0.08802854536726222)

pred_mlj = [1 if x>0.5 else 0 for x in pred_proba_mlj]
print("mse mlj:",mean_squared_error(y,pred_proba_mlj))
# ('mse mlj:', 0.030547386604540198)

# mljar_fit_params = {'max_features': 0.7, 'min_samples_split':  40, 'criterion': "entropy", 'min_samples_leaf': 20}
mljar_fit_params = clf_mlj.selected_algorithm.params['model_params']['fit_params']
print("mljar_fit_params", mljar_fit_params)

# several options of seeds
# Not sure which one is used by mljar
# Based on trial/error, chose 2016 for the constructor and cv_state for the train_test_split
# Documented in diary/_posts/2017-06-29....md
random_seed = [
  2016,
  clf_mlj.selected_algorithm.params['random_seed'],
  clf_mlj.selected_algorithm.params['train_params']['cv_state'],
  None
]

########################
print("Random forest with same params")
i=0
j=2
clf_skl = RandomForestClassifier(
  n_estimators = 5,
  criterion = mljar_fit_params['criterion'],
  max_features = mljar_fit_params['max_features'],
  min_samples_split = mljar_fit_params['min_samples_split'],
  min_samples_leaf = mljar_fit_params['min_samples_leaf'],
  random_state=random_seed[i]
)
# clf_skl.fit(X, y)
# clf_skl.fit(X[0:95,], y[0:95,])
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.95, stratify=y, random_state=random_seed[j])
clf_skl.fit(X_train, y_train)
pred_proba_skl = clf_skl.predict_proba(X)
pred_proba_skl = pred_proba_skl[:,pred_proba_skl.shape[1]-1]
print("mse(skl,mlj)[%i,%i]*1000 = %0.4f" % (i, j, 1000*mean_squared_error(pred_proba_mlj,pred_proba_skl)) )
# mse(skl,mlj)[0,2]*1000 = 1.1445

print("pred_proba_skl",pred_proba_skl) # shows probabilities between 0 and 1, with plenty of values being 0 or 1
print("log loss skl:",log_loss(y,pred_proba_skl))
# ('log loss skl:', 0.083311532661261858)

pred_skl = clf_skl.predict(X)
print("pred_skl",pred_skl) # shows values = 0 or 1
print("same",(pred_skl==y).all()) # returns False
print("mse skl:",mean_squared_error(y,pred_skl))
# ('mse skl:', 0.050000000000000001)

print(np.matrix([pred_proba_mlj,pred_proba_skl,y]).transpose())
# Differences below depend on the random_state used in train_test_split
#
# Run 2017-06-28
# [[ 1.          1.          1.        ]
#  [ 0.03076923  0.          0.        ] <<<<<< not exact same
#  [ 0.          0.          0.        ]
#  [ 0.          0.          0.        ]
#  [ 1.          1.          1.        ]
#  [ 1.          1.          1.        ]
#  [ 0.          0.          0.        ]
#  [ 1.          1.          1.        ]
#  [ 0.60767399  0.5247619   1.        ] <<<<<< not exact same
#  [ 0.          0.          0.        ]
#  [ 0.95        0.9         1.        ] <<<<<< not exact same
#  ...
#
# Re-Run 2017-06-29
# [[ 0.94425351  0.95335609  1.        ]
#  [ 0.00888889  0.00787037  0.        ]
#  [ 0.00888889  0.00787037  0.        ]
#  [ 0.00888889  0.00787037  0.        ]
#  [ 0.94425351  0.95335609  1.        ]
#  [ 0.94425351  0.95335609  1.        ]
#  [ 0.00888889  0.00787037  0.        ]
#  [ 0.94425351  0.95335609  1.        ]
#  [ 0.75722403  0.56548178  1.        ]
#  [ 0.00888889  0.00787037  0.        ]
#  [ 0.94425351  0.95335609  1.        ]
# ...
