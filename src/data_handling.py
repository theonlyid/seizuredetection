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
            '../../../data/data_5sec_100Hz_bipolar/', Nsub=14)

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
        blabel_string = list(compress(self.label_strings, blabel))

        return blabel_string

    def plot_channels(self, data_array):
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
        Performs a STFT on the entire dataset to get mean spectrogram across trials.
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
        This is where the magic happens!
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

    def generate_features(self, data, y_label, compress_data=True):
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
                            np.mean(data[:,  3:8, tbin, trial], 1).ravel(),
                            np.mean(data[:, 9:27, tbin, trial], 1).ravel()])

            data_array = np.log(np.reshape(data_array, (-1, 18)))

        else:

            for trial in range(data.shape[-1]):       # Each trial
                for tbin in range(data.shape[-2]):    # Each timebin
                    data_array = np.append(data_array,
                                           [data[:, :27, tbin, trial].ravel()])

            data_array = np.log(np.reshape(data_array, (-1, 27*6)))

        X = data_array
        y = np.ones(data_array.shape[0])*y_label

        return X, y

    def generate_dataset(self, normalize=True, multiclass=False):
        idx_null = [idx for idx in range(self.labels.shape[0])if self.labels[idx, 0] > 0]
        idx_bckg = [idx for idx in range(self.labels.shape[0]) if self.labels[idx, 6] > 0]
        np.random.shuffle(idx_bckg)
        idx_bckg = idx_bckg[:800]
        idx_gnsz = [idx for idx in range(self.labels.shape[0]) if self.labels[idx, 9] > 0]
        idx_cpsz = [idx for idx in range(self.labels.shape[0]) if self.labels[idx, 11] > 0]
        idx_tcsz = [idx for idx in range(self.labels.shape[0]) if self.labels[idx, 15] > 0]

        # idx_bl = [idx for idx in range(dh.labels.shape[0]) if dh.labels[idx, baseline_label] > 0]
        # print("Using label {} as baseline".format(baseline_label))

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
            X_bckg, y_bckg = self.generate_features(self.bckg_stft_norm, 1)
            X_gnsz, y_gnsz = self.generate_features(self.gnsz_stft_norm, 2)
            X_cpsz, y_cpsz = self.generate_features(self.cpsz_stft_norm, 3)
            X_tcsz, y_tcsz = self.generate_features(self.tcsz_stft_norm, 4)

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

        return X, y

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

        cores = multiprocessing.cpu_count()

        clf = svm.SVC(kernel='rbf', C=1, gamma='scale')
        # scorer = make_scorer(matthews_corrcoef)

        cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        print("\n\n Training svm on dataset")
        scores = cross_val_score(clf, X, y, cv=cv, scoring='balanced_accuracy', n_jobs=cores)
        print("\n\n cross-validation accuracy: %0.2f (+/- %0.2f CI)" % (scores.mean(), scores.std()*2))

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

    def simulate(self):
        """
        Runs a simulation of the data processing pipeline. This demonstrates the processflow.
        """

        # fig, axes = plt.subplots(3, 2)

        # for ax, ch in zip(axes.ravel(), range(6)):
        #     plt.subplot(ax)
        #     plt.plot(self.data[ch, :, 0])

        # print("Figure generated.\n")

        # input("Press any key to continue...")

        # plt.show()

        # Train a multilayer perceptron on the dataset
        self.X_full, self.y_full = self.generateFullFeatures()
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_full, self.y_full, test_size=0.2)
        self.scores_svm_full, self.clf_full = self.classifySVM(self.X_full, self.y_full)

        try:
            self.scores_mlp_full, self.mlp_full = self.classifyMLP(self.X_full, self.y_full)
            print("\n\nTraining accuracy reported by neural net = %0.2f" % (self.mlp_full.score(X_train, y_train)))
            print("\n\nTest accuracy = %0.2f" % (self.mlp_full.score(X_test, y_test)))
            self.y_pred_full = self.mlp_full.predict(X_test)


        except MemoryError as e:
            print("Out of Memory! Dataset too large!")
            sys.exit()

        else:
            pass
        finally:
            pass

        # Too good to be true?



        self.X, self.y = self.generate_dataset(normalize=True, multiclass=False)
        self.scores_svm_less, self.clf_less = self.classifySVM(self.X, self.y)
        self.scores_mlp_less, self.mlp_less = self.classifyMLP(self.X, self.y)

        # print("Balanced accuracy for svm: %0.1f%%, mlp: %0.1f%%" % (self.scores_svm*100, self.scores_mlp*100))


if __name__ == '__main__':

    dh = data_handling()
    dh.load_data()
    dh.simulate()
