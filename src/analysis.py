#%% [markdown]
# # Classifying seizures using spectral contrasting
#
# Author: Ali Zaidi
# 
# ### BRIEF INTRODUCTION
# 
# The objective of this analysis is to demonstrate the usefulness of smart
# feature engineering. The methods have already been defined in src/data_handling.py.


#%%
# Import all dependencies
from sklearn import svm
import src.data_loader as dl
import time
import numpy as np
from itertools import compress
import matplotlib
import matplotlib.pyplot as plt
import tqdm
from scipy import signal
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.neural_network import MLPClassifier
import multiprocessing
import scipy.stats
from sklearn.manifold import TSNE
import sys
import mpld3

#%%
# Import the data handling script and initialize the data_handling object.
import src.data_handling
dh = src.data_handling.data_handling()
dh.load_data()

#%% [markdown]
# ## Understanding the data
#
# The data has been divided into the following objects:
#
# data:     timeseries data from 6 EEG channels
#
# labels:   a binary array identifying the epoch label
# 
#%%
# The dataset is arranged as an array with shape channels x timeseries x epochs.
# Let's have a look at the shape of our data object. 
print(dh.data.shape)

#%%
# There are 8000 epochs of 6 channels sampled at 500 Hz.
# Let's plot the first epoch:
plot, axes = plt.subplots(3, 2)

for ax, ch in zip(axes.ravel(), range(6)):
    plt.subplot(ax)
    plt.plot(np.linspace(0, 999, num=500), dh.data[ch, :, 0])
    plt.xlabel('time (ms)')

#%%
# The label epoch represents normal EEG data
print(dh.get_label_string(dh.labels[0]))

#%% 
X, y = dh.generate_dataset(normalize=False, multiclass=False)
scores_svm_less, clf_less = dh.classifySVM(X, y)
scores_mlp_less, mlp_less = dh.classifyMLP(X, y)

#%%
# Train a multilayer perceptron on the dataset
X_full, y_full = dh.generateFullFeatures()
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2)
scores_svm_full, clf_full = dh.classifySVM(X_full, y_full)

try:
    scores_mlp_full, mlp_full = dh.classifyMLP(X_full, y_full)
    print("\n\nTraining accuracy reported by neural net = %0.2f" % (mlp_full.score(X_train, y_train)))
    print("\n\nTest accuracy = %0.2f" % (mlp_full.score(X_test, y_test)))
    y_pred_full = mlp_full.predict(X_test)

except MemoryError as e:
    print("Out of Memory! Dataset too large!")
    sys.exit()
