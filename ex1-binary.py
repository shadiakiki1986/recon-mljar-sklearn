"""
Reconciling results from MLJAR and sklearn.
  Random forest classifier
  X==y


Usage
  pew new --python=python2 MLJAR_TEST
  pip install mljar sklearn numpy scipy pandas
  export MLJAR_TOKEN=exampleexampleexample

Author: Shadi Akiki

Published here:
  https://gist.github.com/shadiakiki1986/f4bec0b3d86d41c143013a0e9770144d
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
from mljar import Mljar

#################################
# Data

n_samples = 100
# Using n_samples =  20 yields an error in MLJAR fitting
# Using n_samples =  50 yields prediction = 0.4 for all points from the MLJAR prediction if the best model uses min_samples_split = 50
# Using n_samples = 100 yields prediction = 0.45 or 0.55 for all points from the MLJAR prediction
#   but coincidentally, the best model uses min_samples_split = 4

half = n_samples / 2

# Very simple
# Binary classification
# input == output
X = np.zeros((n_samples,1))
X[ 0:half,0] = 2
# X[half:n_samples,1] = 1
y = X[:,0]

print("X",X.transpose())
print("y",y)

########################
print("MLJAR")
clf_mlj = Mljar(
  project='Recon mljar-sklearn',
  # experiment='Experiment 1', # for n_samples = 50
  # experiment='Experiment 2', # for n_samples = 100, and 2 classes: 0, 1
  # experiment='Experiment 3', # for n_samples = 100, and 2 classes: 0, 2
  experiment='Experiment 4', # for n_samples = 100, and 2 classes: 0, 2, and just re-run anew for probabilities
  metric='auc',
  algorithms=['rfc'],
  validation_kfolds=None,
  validation_shuffle=False,
  validation_stratify=True,
  validation_train_split=0.05,
  tuning_mode='Normal', # Used Sport for experiments 1-3
  create_ensemble=False,
  single_algorithm_time_limit='1'
)

print("fit")
clf_mlj.fit(X,y) #,dataset_title="Ones and zeros")

print("predict")
pred_mlj = clf_mlj.predict(X)
pred_mlj = pred_mlj.squeeze().values
print("pred_mlj",pred_mlj) # shows values = 0.45 or 0.55
print("same",(pred_mlj==y).all()) # returns False

# mljar_fit_params = {'max_features': 0.5, 'min_samples_split': 50, 'criterion': "gini",    'min_samples_leaf': 1}
# mljar_fit_params = {'max_features': 0.7, 'min_samples_split':  4, 'criterion': "entropy", 'min_samples_leaf': 2}
mljar_fit_params = clf_mlj.selected_algorithm.params['model_params']['fit_params']
print("mljar_fit_params", mljar_fit_params)

########################
print("Random forest with same params")
clf_skl = RandomForestClassifier(
  max_features = mljar_fit_params['max_features'],
  min_samples_split = mljar_fit_params['min_samples_split'],
  criterion = mljar_fit_params['criterion'],
  min_samples_leaf = mljar_fit_params['min_samples_leaf']
)
clf_skl.fit(X, y)
pred_skl = clf_skl.predict(X)

print("pred_skl",pred_skl) # shows values = 0 or 2
print("same",(pred_skl==y).all()) # returns True

pred_proba_skl = clf_skl.predict_proba(X)
print("pred_proba_skl",pred_proba_skl) # shows values = 0 or 1

