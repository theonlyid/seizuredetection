#! /usr/bin/env python
"""
Bundle of methods for handling Temple Unversity EEG corpus data

Requirements: data_laoder.py from Gagan Narula
(ensure its in your current folder or add it to path)

Author: Ali Zaidi
Version: 0.1
Date:   14.11.2018
(c) All Rights Reserved
"""

from __future__ import print_function, absolute_import
import src.data_loader as dl
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE
import multiprocessing
from scipy.ndimage import maximum_filter, generate_binary_structure


class data_handling:
    """
    Handles the loading and processing of temple-university data.

    For understanding process flow see simulate().
    """

    def __init__(self):

        # self.load_data()

        self.nperseg = 64
        self.noverlap = 3 * np.ceil(self.nperseg / 4)
        self.fmax = 50

    def load_data(self):
        """
        Loads the data and squeezes the arrays for easier handling.
        """
        print("Loading dataset...")
        # Load the dataset
        subIDs, data, labels = dl.load_processed_data_N_subjects_allchans(
            'data/', Nsub=14)

        if len(data) > 1:

            # If more than one patient loaded, append data to single array
            data_arr = np.array(data[0])
            label_arr = labels[0]

            for sub in range(1, len(data)):
                data_arr = np.append(data_arr, data[sub], axis=1)
                label_arr = np.append(label_arr, labels[sub], axis=0)

        else:
            # Remove the extra dimension at axis=0
            data_arr = np.squeeze(data)
            label_arr = np.squeeze(labels)

        # Move trials to the end so data array is 'nchan x timeseries x trials'
        self.data = np.moveaxis(data_arr, 1, -1)
        self.labels = np.array(label_arr)

        self.label_strings = dl.available_stringlabels

        valid_indices = np.sum(self.labels, axis=0)
        names = [[self.label_strings[i], i, valid_indices[i]] for i in range(len(valid_indices)) if valid_indices[i] > 0]
        print("A summary of valid labels is below: \nFormat: [Label name, label index, Label count]")
        for i in range(len(names)):
            print(names[i])
        return

    def get_label_string(self, index_list):
        """
        Returns the string for a boolean list for the label.

        See self.labels[0] for reference
        """

        blabel = [bool(x) for x in index_list]
        blabel_string = list(np.squeeze(self.label_strings, blabel))

        return blabel_string

    def plot_epoch(self, data_array):
        """
        Plots the 6 channels in a subplot for visualization. Takes an image in the shape nchan x timestamps.
        """

        plt.figure()
        for p in range(1, 7):
            plt.subplot(6, 1, p)
            plt.plot(data_array[p-1, :])

        plt.draw()
        plt.show()
        return

    def get_norm_array(self, data):
        """
        Performs a STFT on the entire dataset to get mean power for each frequency across trials.
        This is used to normailze the data and baseline arrays for feature extraction.
        """

        # NOTE: THIS NEEDS TO BE ADAPTED IN VERSION 0.2. It should return normalized arrays.

        f, _, data_stft = signal.stft(
            data, fs=100, nperseg=self.nperseg, noverlap=self.noverlap, axis=1)

        # Average across trials and then timebins
        data_stft_mean = np.mean(np.abs(data_stft), axis=-1)
        norm_array = np.mean(np.abs(data_stft_mean), axis=-1)

        # Repleat matrix across timebins to cast it in the same shape as data and baseline arrays. See calculate_snr()
        norm_array = np.repeat(norm_array[:,:,None], data_stft.shape[3], axis=2)

        return norm_array

    def get_stft(self, data_array, norm_array=[], normalize=True):
        """
        
        """

        # Calculate the short-time fourrier transform
        # f, _, baseline_stft = signal.stft(
        #     baseline_array, fs=100, nperseg=self.nperseg, noverlap=self.noverlap, axis=1)
        f, _, data_stft = signal.stft(
            data_array, fs=100, nperseg=self.nperseg, noverlap=self.noverlap, axis=1)

        # Make last axis as trials
        # baseline_stft = np.moveaxis(np.abs(baseline_stft), 2, 3)
        data_stft = np.moveaxis(np.abs(data_stft), 2, 3)

        if normalize:

            # Facilitate vectorized division
            norm_array_data = np.repeat(norm_array[:, :, :, None], data_stft.shape[3], axis=3)
            # norm_array_bl = np.repeat(norm_array[:, :, :, None], baseline_stft.shape[3], axis=3)

            # Normalize the spectrograms for calculating SNR
            data_stft_norm = data_stft / norm_array_data
            # baseline_stft_norm = baseline_stft / norm_array_bl

            return data_stft_norm, f

        else:

            return data_stft, f

    def get_bands(self, data_array_norm, baseline_array_norm, f):
        """
        Obtains the range of bands where the SNR is the highest.
        """

        fmax = 50
        fidx = f < fmax
        fnum = f[fidx].size

        band_tot = np.zeros((fnum, fnum, data_array_norm.shape[0], data_array_norm.shape[2], data_array_norm.shape[3]))
        band_tot_bl = np.zeros((fnum, fnum, baseline_array_norm.shape[0], baseline_array_norm.shape[2], baseline_array_norm.shape[3]))
        for i in range(fnum):
            for j in range(fnum):
                if j > i:
                    idx = (f >= f[i]) & (f < f[j])
                    band_tot[i, j, :, :] = np.sum(data_array_norm[:, idx, :, :], axis=1) / (f[j] - f[i])
                    band_tot_bl[i, j, :, :] = np.sum(baseline_array_norm[:, idx, :, :], axis=1) / (f[j] - f[i])


        band_tot_bl1 = np.mean(band_tot_bl, axis=3)     # average across time bins
        band_tot_bl2 = np.repeat(band_tot_bl1[:, :, :, None, :], band_tot_bl.shape[3], axis=3)    # repeat same value across time
        return band_tot, band_tot_bl2, f[fidx]

    def get_snr(self, target, baseline):
        """
        Returns the SNR given two vectors: target and baseline.
        """

        mu_cue = np.mean(target, axis=-1)     # average accross trials
        mu_bl = np.mean(baseline, axis=-1)     # average accross trials
        std_cue = np.std(target, axis=-1)     # average accross trials
        std_bl = np.std(baseline, axis=-1)     # average accross trials

        snr = np.abs((mu_cue - mu_bl) / (std_cue + std_bl))
        snr2 = np.nanmax(snr, axis=-1)
        if snr2.shape[0] == snr.shape[1]:
            snr2 = np.nanmax(snr2, axis=-1)

        return snr2

    # TODO: Finish method to automatically find optimal bands from STFT frames
    # =============================================================================
    def get_snr_maxima(self, snr):
        """
        This function returns the local maxima (in a 2x2 grid) from the SNR matrix
        """

        neighborhood = generate_binary_structure(2,2)
        snr_maxima = maximum_filter(snr, neighborhood = neighborhood) == snr

        # Get the indices of the local maxima
        idx = np.where(snr_maxima==True)

        # Get the SNR values for the local maxima
        snr_vals = snr[idx]
        
        # And turn them into an array with indices and values
        snr_array = np.append(idx, snr_vals, axis=1)

        return snr_array

    # =============================================================================


    def butter_filter(self, data, low_pass, high_pass, fs, order=10):
        """
        Generates a 10th order butterworth filter and performs filtfilt on the signal.
        """

        nyq = fs/2
        low = low_pass/nyq
        high = high_pass/nyq

        b, a = signal.butter(order, [low, high], btype='band')
        filt_data = np.abs(signal.hilbert(signal.filtfilt(b, a, data, axis=1), axis=1))
        return filt_data

    def plot_snr(self, snr):
        plt.figure()
        plt.imshow(snr, origin='lower', interpolation='bilinear')
        plt.colorbar()
        plt.clim(0,)
        plt.xlabel("Band start (Hz)")
        plt.ylabel("Band end (Hz)")
        plt.title("SNR")
        plt.draw()
        plt.show()

    def generate_features(self, data, y_label, compress_data=True, log_transform=True):
        """
        Generates feature vectors for feeding into SVM.

        Currently, that means taking the mean power in three frequency ranges:
        (0 - 3 Hz, 3 - 12 Hz, 12 - 30 Hz) generating 18 in all (nchans = 6)

        Inputs:
            data_array: array with shape nchan x f x tbin x trials (see get_norm_array())
            y_label     label to be given (used to generate vector y)

        Returns:
            X:  Array of features x trials (different from number of epochs)
            y:  vector with class label
        """

        # For each STFT timebin, divide data into three bins and get mean power
        data_array = np.array([])
        # bl_array = np.array([])

        if compress_data:

            for trial in range(data.shape[-1]):
                for tbin in range(data.shape[-2]):    # Each timebin
                    data_array = np.append(
                        data_array, [
                            np.mean(data[:,   :2, tbin, trial], 1).ravel(),
                            np.mean(data[:,  3:9, tbin, trial], 1).ravel(),
                            np.mean(data[:, 9:27, tbin, trial], 1).ravel()])

            if log_transform:
                data_array = np.log(np.reshape(data_array, (-1, 18)))
            else:
                data_array = np.reshape(data_array, (-1, 18))

        else:

            for trial in range(data.shape[-1]):       # Each trial
                for tbin in range(data.shape[-2]):    # Each timebin
                    data_array = np.append(data_array,
                                           [data[:, :27, tbin, trial].ravel()])

            data_array = np.log(np.reshape(data_array, (-1, 27*6)))

        X = data_array
        y = np.ones(data_array.shape[0])*y_label

        return X, y

    def generate_features_from_snr(self, data, y_label, compress_data=True, log_transform=True):
        """
        Generates feature vectors for feeding into SVM.

        Currently, that means taking the mean power in three frequency ranges:
        (0 - 3 Hz, 3 - 12 Hz, 12 - 30 Hz) generating 18 in all (nchans = 6)

        Inputs:
            data_array: array with shape nchan x f x tbin x trials (see get_norm_array())
            y_label     label to be given (used to generate vector y)

        Returns:
            X:  Array of features x trials (different from number of epochs)
            y:  vector with class label
        """

        # For each STFT timebin, divide data into three bins and get mean power
        data_array = np.array([])
        # bl_array = np.array([])

        for trial in range(data.shape[-1]):
            for tbin in range(data.shape[-2]):    # Each timebin
                data_array = np.append(
                    data_array, [
                        np.mean(data[:,   :2, tbin, trial], 1).ravel(),
                        np.mean(data[:,  2:9, tbin, trial], 1).ravel(),
                        np.mean(data[:, 9:20, tbin, trial], 1).ravel(),
                        np.mean(data[:, 20:30, tbin, trial], 1).ravel(),
                        np.mean(data[:, 30:,   tbin, trial], 1).ravel()])

        if log_transform:
            data_array = np.log(np.reshape(data_array, (-1, 30)))
        else:
            data_array = np.reshape(data_array, (-1, 30))

        data_array = np.log(np.reshape(data_array, (-1, 27*6)))

        X = data_array
        y = np.ones(data_array.shape[0])*y_label

        return X, y

    def visualize_data(self):
        signal.butter()

    def generate_dataset(self, normalize=True, multiclass=False):
        idx_null = [idx for idx in range(self.labels.shape[0])if self.labels[idx, 0] > 0]
        idx_bckg = [idx for idx in range(self.labels.shape[0]) if self.labels[idx, 6] > 0]
        np.random.shuffle(idx_bckg)
        idx_bckg = idx_bckg[:800]
        idx_gnsz = [idx for idx in range(self.labels.shape[0]) if self.labels[idx, 9] > 0]
        idx_cpsz = [idx for idx in range(self.labels.shape[0]) if self.labels[idx, 11] > 0]
        idx_tcsz = [idx for idx in range(self.labels.shape[0]) if self.labels[idx, 15] > 0]

        if normalize:
            print("Generating normalized dataset...")
            norm = self.get_norm_array(self.data)
            self.null_stft_norm, _ = self.get_stft(self.data[:, :, idx_null], norm)
            self.bckg_stft_norm, _ = self.get_stft(self.data[:, :, idx_bckg], norm)
            self.gnsz_stft_norm, _ = self.get_stft(self.data[:, :, idx_gnsz], norm)
            self.cpsz_stft_norm, _ = self.get_stft(self.data[:, :, idx_cpsz], norm)
            self.tcsz_stft_norm, f = self.get_stft(self.data[:, :, idx_tcsz], norm)

        else:
            print("Generating dataset...")
            norm = self.get_norm_array(self.data)
            self.null_stft_norm, _ = self.get_stft(self.data[:, :, idx_null], norm)
            self.bckg_stft_norm, _ = self.get_stft(self.data[:, :, idx_bckg], norm)
            self.gnsz_stft_norm, _ = self.get_stft(self.data[:, :, idx_gnsz], norm)
            self.cpsz_stft_norm, _ = self.get_stft(self.data[:, :, idx_cpsz], norm)
            self.tcsz_stft_norm, f = self.get_stft(self.data[:, :, idx_tcsz], norm)

        if multiclass:
            print("Generating training datasets for data...")
            X_null, y_null = self.generate_features(self.null_stft_norm, 0)
            X_bckg, y_bckg = self.generate_features(self.bckg_stft_norm, 0)
            X_gnsz, y_gnsz = self.generate_features(self.gnsz_stft_norm, 1)
            X_cpsz, y_cpsz = self.generate_features(self.cpsz_stft_norm, 2)
            X_tcsz, y_tcsz = self.generate_features(self.tcsz_stft_norm, 3)

        else:
            print("Generating training datasets for data...\n")
            X_null, y_null = self.generate_features(self.null_stft_norm, 0)
            X_bckg, y_bckg = self.generate_features(self.bckg_stft_norm, 0)
            X_gnsz, y_gnsz = self.generate_features(self.gnsz_stft_norm, 1)
            X_cpsz, y_cpsz = self.generate_features(self.cpsz_stft_norm, 1)
            X_tcsz, y_tcsz = self.generate_features(self.tcsz_stft_norm, 1)

        # Append the matrices
        X = np.append(X_null, X_bckg, axis=0)
        X = np.append(X, X_gnsz, axis=0)
        X = np.append(X, X_cpsz, axis=0)
        X = np.append(X, X_tcsz, axis=0)

        y = np.append(y_null, y_bckg, axis=0)
        y = np.append(y, y_gnsz, axis=0)
        y = np.append(y, y_cpsz, axis=0)
        y = np.append(y, y_tcsz, axis=0)


        ds = np.empty((X.shape[0], X.shape[1]+1));
        ds[:,:-1] = X
        ds[:,-1] = y


        return ds[:,:-1], ds[:,-1].astype(int)

    def generateFullFeatures(self):
        idx_null = [idx for idx in range(self.labels.shape[0])if self.labels[idx, 0] > 0]
        idx_bckg = [idx for idx in range(self.labels.shape[0]) if self.labels[idx, 6] > 0]
        np.random.shuffle(idx_bckg)
        idx_bckg = idx_bckg[:800]
        idx_gnsz = [idx for idx in range(self.labels.shape[0]) if self.labels[idx, 9] > 0]
        idx_cpsz = [idx for idx in range(self.labels.shape[0]) if self.labels[idx, 11] > 0]
        idx_tcsz = [idx for idx in range(self.labels.shape[0]) if self.labels[idx, 15] > 0]

        X_full = np.append(self.data[:, :, idx_null].ravel(), self.data[:, :, idx_bckg].ravel(), axis=0)
        X_full = np.append(X_full, self.data[:, :, idx_gnsz].ravel(), axis=0)
        X_full = np.append(X_full, self.data[:, :, idx_cpsz].ravel(), axis=0)
        X_full = np.append(X_full, self.data[:, :, idx_tcsz].ravel(), axis=0)

        X_full = np.reshape(X_full, (-1, 3000))

        y_full = np.append(
            np.zeros((len(idx_null)+len(idx_bckg),), dtype=np.int64),
            np.ones((len(idx_gnsz)+len(idx_cpsz)+len(idx_tcsz),), dtype=np.int64))

        return X_full, y_full

    def timerfunc(func):
        """
        A timer decorator to assess time taken by different classifiers.
        """
        def function_timer(*args, **kwargs):
            """
            A nested function for timing other functions
            """
            start = time.time()
            value = func(*args, **kwargs)
            end = time.time()
            runtime = end - start
            msg = "The runtime for {func} took {time} seconds to complete"
            print(msg.format(func=func.__name__,
                             time=runtime))
            return value
        return function_timer

    @timerfunc
    def classifySVM(self, X, y, return_classifer=False):
        """
        Trains an SVM on the data.
        """

        ds = np.append(X, y, axis=1)
        np.random.seed(42)
        np.random.shuffle(ds)

        cores = multiprocessing.cpu_count()

        clf = SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma='scale',
            kernel='rbf', max_iter=-1, probability=True, random_state=42,
            shrinking=True, tol=0.001, verbose=0)
        
        cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        print("Performing 5x5 cross-validation on dataset")
        scores = cross_val_score(clf, X, y, cv=cv, scoring='balanced_accuracy', n_jobs=cores)
        print("cross-validation accuracy: %0.2f (+/- %0.2f CI)" % (scores.mean(), scores.std()*2))

        return scores, clf

    @timerfunc
    def classifyMLP(self, X, y, scale_units=False):
        """
        Trains a MLP on the data
        """

        if scale_units:
            n_features = 2 * X.shape[1]
        else:
            n_features = 50

        print("\n\nTraining neural network on dataset")

        scores = np.array([])
        for k in range(5):

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2)

            mlp = MLPClassifier(
                hidden_layer_sizes=(n_features, 2*n_features, n_features), alpha=1e-4,
                solver='sgd', verbose=0, tol=1e-4, random_state=1,
                learning_rate_init=.1, max_iter=200)

            mlp.fit(X_train, y_train)
            y_pred = mlp.predict(X_test)
            scores = np.append(scores, matthews_corrcoef(y_pred, y_test))

        print("\n \nncross-validation accuracy: %0.2f (+/- %0.2f CI)" % (scores.mean(), scores.std()*2))
        return scores, mlp

    def predict_epoch(self, epoch, clf, norm):
        epoch_stft_norm, _ = self.get_stft(epoch[:,:,np.newaxis], norm)
        x, _ = self.generate_features(epoch_stft_norm, 0)

        y_pred = clf.predict(x)

        return np.median(y_pred).astype(int)

    def plot_confusion_matrix(self, cm, normalize=True):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        classes=list(np.arange(4))
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               title='Normalized confusion matrix',
               ylabel='True label',
               xlabel='Predicted label')
        ax.set(ylim = [-0.5, 3.5])
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        ax.set(ylim=[-0.5, 3.5], xlim=[-0.5,3.5])
        fig.tight_layout()
        return ax

    def simulate(self):
        """
        Runs a simulation of the data processing and classification  pipeline.
        """

        # Load the data
        self.load_data()

        # Make epochs the first axis to feed to train_test_split()
        data = np.moveaxis(self.data, -1, 0)

        # Init SVM object
        clf = SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
          decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
          max_iter=-1, probability=True, random_state=42, shrinking=True, tol=0.001,
          verbose=0)

        # Create integer labels for data classes
        y_labels = np.empty((self.labels.shape[0], 1))
        idx_null = [idx for idx in range(self.labels.shape[0]) if self.labels[idx, 0] > 0]
        idx_bckg = [idx for idx in range(self.labels.shape[0]) if self.labels[idx, 6] > 0]
        idx_gnsz = [idx for idx in range(self.labels.shape[0]) if self.labels[idx, 9] > 0]
        idx_cpsz = [idx for idx in range(self.labels.shape[0]) if self.labels[idx, 11] > 0]
        idx_tcsz = [idx for idx in range(self.labels.shape[0]) if self.labels[idx, 15] > 0]
        y_labels[idx_null] = 0
        y_labels[idx_bckg] = 0
        y_labels[idx_gnsz] = 1
        y_labels[idx_cpsz] = 2
        y_labels[idx_tcsz] = 3

        # Initialize variables for epoch-level cross-validation
        n_cv = 5
        f1score = np.empty((n_cv,))
        scores = np.empty((n_cv,))
        cc = np.empty((4,4,n_cv))

        for cv in range(n_cv):
            print("\n\nRunning CV-fold {} out of {}".format(cv+1, n_cv))
            X_train, X_test, y_train, y_test = train_test_split(data, y_labels, stratify=y_labels, test_size=0.2, random_state=42*cv)
        
            X_train = np.moveaxis(X_train, 0, -1 )
        
            norm = self.get_norm_array(X_train)
        
            X_train_stft, _ = self.get_stft(X_train, norm_array=norm)
        
            idx_0 = [idx for idx in range(len(y_train)) if y_train[idx]==0]
            idx_1 = [idx for idx in range(len(y_train)) if y_train[idx]==1]
            idx_2 = [idx for idx in range(len(y_train)) if y_train[idx]==2]
            idx_3 = [idx for idx in range(len(y_train)) if y_train[idx]==3]
            
            print("Generating features...")
            # Note that for Label-0 we are using Reduced number of epochs to facilitate faster execution times
            X_0, y_0 = self.generate_features(X_train_stft[:,:,:,idx_0[:len(idx_1)+len(idx_2)+len(idx_3)]], 0)
            X_1, y_1 = self.generate_features(X_train_stft[:,:,:,idx_1], 1)
            X_2, y_2 = self.generate_features(X_train_stft[:,:,:,idx_2], 2)
            X_3, y_3 = self.generate_features(X_train_stft[:,:,:,idx_3], 3)
        
            X = np.append(X_0, X_1, axis=0)
            X = np.append(X, X_2, axis=0)
            X = np.append(X, X_3, axis=0)
        
            y = np.append(y_0, y_1, axis=0)
            y = np.append(y, y_2, axis=0)
            y = np.append(y, y_3, axis=0)
        
            ds = np.empty((X.shape[0], X.shape[1]+1));
            ds[:,:-1] = X
            ds[:,-1] = y
        
            np.random.shuffle(ds)
            
            print("Training classifier...")
            clf.fit(ds[:,:-1], ds[:,-1])
        
            X_test = np.moveaxis(X_test, 0, -1)
        
            y_pred = np.empty((len(y_test),1))
            for i in range(len(y_test)):
                y_pred[i] = self.predict_epoch(X_test[:,:,i], clf, norm)
        
        
            scores[cv] = balanced_accuracy_score(y_test, y_pred)
        
            cc[:,:,cv] = confusion_matrix(y_test, y_pred)
            
            f1score[cv] = f1_score(y_test, y_pred, average='weighted')
        
        print("mean F1 scores are %0.3f +/- %0.3f" %(f1score.mean(), f1score.std()))
        print("mean balanced-accuracy scores are %0.3f +/- %0.3f" %(scores.mean(), scores.std()))

        # Generate the average confusion matrix and draw the plot
        CV = np.sum(cc, axis=-1)
        CV1 = CV/np.sum(CV, axis=1)[:,np.newaxis]
        self.plot_confusion_matrix(CV1)


if __name__ == '__main__':

    dh = data_handling()
    dh.simulate()
