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

# Until https://github.com/mljar/mljar-api-python/issues/2
# gets fixed,
# manually get the result to use in the prediction
# by browsing mljar.com
from mljar.client.result import ResultClient
client = ResultClient(clf_mlj.project.hid)
results = client.get_results(clf_mlj.experiment.hid)
# len(results)
# results = client.get_results(None)
# len(results)
# rid = 'RMxezRO0eaGy'
rid = raw_input("Enter result ID from mljar.com (until issue#2 is solved)")
selected = [x for x in results if x.hid==rid]
if len(selected)!=1:
  raise Exception("len(selected)!=1")

clf_mlj.selected_algorithm = selected[0]


print("predict")
pred_proba_mlj = clf_mlj.predict(X)
pred_proba_mlj = pred_proba_mlj.squeeze().values
print("pred_proba_mlj",pred_proba_mlj) # shows probabilities
print("log loss mlj:",log_loss(y,pred_proba_mlj))


pred_mlj = [1 if x>0.5 else 0 for x in pred_proba_mlj]
print("mse mlj:",mean_squared_error(y,pred_proba_mlj))


# mljar_fit_params = {'max_features': 0.5, 'min_samples_split': 50, 'criterion': "gini",    'min_samples_leaf': 1}
# mljar_fit_params = {'max_features': 0.7, 'min_samples_split':  4, 'criterion': "entropy", 'min_samples_leaf': 2}
mljar_fit_params = clf_mlj.selected_algorithm.params['model_params']['fit_params']
print("mljar_fit_params", mljar_fit_params)

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
clf_skl.fit(X, y)
clf_skl.fit(X[0:95,], y[0:95,])
pred_skl = clf_skl.predict(X)

print("pred_skl",pred_skl) # shows values = 0 or 1
print("same",(pred_skl==y).all()) # returns False

pred_proba_skl = clf_skl.predict_proba(X)
pred_proba_skl = pred_proba_skl[:,pred_proba_skl.shape[1]-1]
print("pred_proba_skl",pred_proba_skl) # shows probabilities between 0 and 1, with plenty of values being 0 or 1
print("log loss skl:",log_loss(y,pred_proba_skl))
print("mse skl:",mean_squared_error(y,pred_skl))

print(np.matrix([pred_proba_mlj,pred_proba_skl,y]).transpose())

#################
#for i in range(1,100):
#  clf_skl.fit(X[0:i,], y[0:i,])
#  pred_proba_skl = clf_skl.predict_proba(X)
#  pred_proba_skl = pred_proba_skl[:,pred_proba_skl.shape[1]-1]
#  print("pred_proba_skl", set(pred_proba_skl))
#  raw_input("press a key")
#
#sub = (0,1,2,3,4,5,50,51,52,53,54,55,56)
#clf_skl.fit(X[sub,], y[sub,])
#pred_proba_skl = clf_skl.predict_proba(X)
#pred_proba_skl = pred_proba_skl[:,pred_proba_skl.shape[1]-1]
#print("pred_proba_skl", set(pred_proba_skl))
