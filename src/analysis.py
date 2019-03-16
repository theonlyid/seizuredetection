#%% [markdown]
# # Classifying seizures using spectral contrasting
#
# Author: Ali Zaidi
# 
# ### BRIEF INTRODUCTION
#
# Our objective is to train a classifier to detect seizures from EEG data.
# We are using a data from the Temple University dataset. 
# The objective of this analysis is to demonstrate the usefulness of smart
# feature engineering. The methods have already been defined in src/data_handling.py.


#%%
# Import all dependencies
from sklearn import svm
import src.data_loader as dl
import time
import numpy as np
from itertools import compress
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.neural_network import MLPClassifier
import multiprocessing
import scipy.stats
from sklearn.manifold import TSNE
import sys
#%%
# Import the data handling scripts and initialize the object.
import src.data_handling
dh = src.data_handling.data_handling()
dh.load_data()

#%%
#  The dataset is arranged as an array with shape channels x timeseries x epochs.
dh.data.shape

# There are 8000 epochs of 6 channels sampled at 500 Hz.
# Let's plot an epoch:
plot, axes = plt.subplots(3, 2)

for ax, ch in zip(axes.ravel(), range(6)):
    plt.subplot(ax)
    plt.plot(dh.data[ch, :, 0])

plot.suptitle(dh.get_label_string(dh.labels[0]))

#%%
# This is an example
