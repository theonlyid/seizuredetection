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

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import numpy as np
    import data_loader as dl
    from itertools import compress
    import matplotlib.pyplot as plt
    from scipy import signal
    from sklearn import svm
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
    from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
    import multiprocessing
    from tqdm import tqdm
    import scipy.stats


class data_handling:
    """
    Handles the loading and processing of temple-university data.

    For understanding process flow see simulate().
    """

    def __init__(self):

        self.load_data()

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
            '../../data_5sec_100Hz_bipolar/', Nsub=14)

        if len(data) > 1:

            # If more than one patient loaded, append data to single array
            data_arr = np.array(data[0])
            label_arr = labels[0]

            for sub in range(1, len(data)):
                data_arr = np.append(data_arr, data[sub], axis=1)
                label_arr = np.append(label_arr, labels[sub], axis=0)

        else:
            # Remove the extra dimension at axis=0
            data_array = np.squeeze(data)
            labels = np.squeeze(labels)

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

    def generate_features(self, data, y_label):
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

        for trial in tqdm(range(data.shape[-1]), ncols=100, desc='Generating matrix for label {0}'.format(y_label)):       # Each trial
            for tbin in range(data.shape[-2]):    # Each timebin
                for ch in range(self.data.shape[0]):
                    data_array = np.append(data_array,[
                                            np.mean(data[ch,   :2, tbin, trial]),
                                            np.mean(data[ch,  3:8, tbin, trial]),
                                            np.mean(data[ch, 9:27, tbin, trial])])

        data_array = np.reshape(data_array, (-1, 18))

        # for trial in range(self.bl_stft_norm.shape[-1]):       # Each trial
        #     for tbin in range(self.bl_stft_norm.shape[-2]):    # Each timebin
        #         for ch in range(self.bl_stft_norm.shape[0]):
        #             bl_array = np.append(bl_array, [
        #                                     np.mean(self.bl_stft_norm[ch,   :2, tbin, trial]),
        #                                     np.mean(self.bl_stft_norm[ch,  3:8, tbin, trial]),
        #                                     np.mean(self.bl_stft_norm[ch, 9:27, tbin, trial])])
        # bl_array = np.reshape(bl_array, (-1, 18))

        X = data_array
        y = np.ones(data_array.shape[0])*y_label

        return X, y

    def classify(self, X, y, multiclass=True):
        """
        Trains an SVM on the data with 5x5 cross-validation.
        """

        cores = multiprocessing.cpu_count()

        if multiclass:
            clf = svm.SVC(kernel='linear', C=1, gamma='scale')

            cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
            print("Training classifier on dataset")
            scores = cross_val_score(clf, X, y, cv=cv, scoring='balanced_accuracy', n_jobs=cores)
            print("cross-validation accuracy: %0.2f (+/- %0.2f CI)" % (scores.mean(), scores.std()*2))

        else:
            clf = svm.SVC(kernel='rbf', C=1, gamma='scale')

            cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
            print("Training classifier on dataset")
            scores = cross_val_score(clf, X, y, cv=cv, scoring='balanced_accuracy', n_jobs=cores)
            print("cross-validation accuracy: %0.2f (+/- %0.2f CI)" % (scores.mean(), scores.std()*2))

        # Use joblib and multiprocessing to make this part run over multiple cores
        # print("Peforming train-test split with 0.2 test size")
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.2, random_state=42)
        # clf.fit(X_train, y_train)
        # #s = clf.score(X_test, y_test)
        # yhat = clf.predict(X_test)
        # s = balanced_accuracy_score(y_test, yhat)
        # c = matthews_corrcoef(y_test, yhat)
        # print("Matthews corr coef = %0.2f, balanced test score = %0.2f" %(c, s))

        return scores

    def simulate(self):
        """
        Runs a simulation of the data processing pipeline. This demonstrates the processflow.
        """

        idx_null = [idx for idx in range(dh.labels.shape[0]) if dh.labels[idx, 0] > 0]
        idx_bckg = [idx for idx in range(dh.labels.shape[0]) if dh.labels[idx, 6] > 0][:800]
        idx_gnsz = [idx for idx in range(dh.labels.shape[0]) if dh.labels[idx, 9] > 0]
        idx_cpsz = [idx for idx in range(dh.labels.shape[0]) if dh.labels[idx, 11] > 0]
        idx_tcsz = [idx for idx in range(dh.labels.shape[0]) if dh.labels[idx, 15] > 0]


        # idx_bl = [idx for idx in range(dh.labels.shape[0]) if dh.labels[idx, baseline_label] > 0]
        # print("Using label {} as baseline".format(baseline_label))

        self.null_stft, _ = self.get_stft(self.data[:, :, idx_null], normalize=False)
        self.bckg_stft, _ = self.get_stft(self.data[:, :, idx_bckg], normalize=False)
        self.gnsz_stft, _ = self.get_stft(self.data[:, :, idx_gnsz], normalize=False)
        self.cpsz_stft, _ = self.get_stft(self.data[:, :, idx_cpsz], normalize=False)
        self.tcsz_stft, _ = self.get_stft(self.data[:, :, idx_tcsz], normalize=False)

        print("Generating training datasets for non-normalized data...")
        X_null, y_null = self.generate_features(self.null_stft, 0)
        X_bckg, y_bckg = self.generate_features(self.bckg_stft, 0)
        X_gnsz, y_gnsz = self.generate_features(self.gnsz_stft, 1)
        X_cpsz, y_cpsz = self.generate_features(self.cpsz_stft, 2)
        X_tcsz, y_tcsz = self.generate_features(self.tcsz_stft, 3)


        # Append the matrices

        X = np.append(X_null, X_bckg, axis=0)
        X = np.append(X, X_gnsz, axis=0)
        X = np.append(X, X_cpsz, axis=0)
        X = np.append(X, X_tcsz, axis=0)

        y = np.append(y_null, y_bckg, axis=0)
        y = np.append(y, y_gnsz, axis=0)
        y = np.append(y, y_cpsz, axis=0)
        y = np.append(y, y_tcsz, axis=0)

        print("Training classifier with 5x5 CV...")
        self.scores = self.classify(X, y, multiclass=True)

        # Vectorize this part of the code
        print("Normalizing data...")
        norm = self.get_norm_array(self.data)
        self.null_stft_norm, _ = self.get_stft(self.data[:, :, idx_null], norm)
        self.bckg_stft_norm, _ = self.get_stft(self.data[:, :, idx_bckg], norm)
        self.gnsz_stft_norm, _ = self.get_stft(self.data[:, :, idx_gnsz], norm)
        self.cpsz_stft_norm, _ = self.get_stft(self.data[:, :, idx_cpsz], norm)
        self.tcsz_stft_norm, f = self.get_stft(self.data[:, :, idx_tcsz], norm)

        # band_tot, band_tot_bl2, f = self.get_bands(self.data_stft_norm, self.bl_stft_norm, f)

        # print("Obtaining SNR...")
        # snr = self.get_snr(band_tot, band_tot_bl2)
        # self.plot_snr(snr)

        print("Generating training datasets for normalized data...")
        X_null, y_null = self.generate_features(self.null_stft_norm, 0)
        X_bckg, y_bckg = self.generate_features(self.bckg_stft_norm, 0)
        X_gnsz, y_gnsz = self.generate_features(self.gnsz_stft_norm, 1)
        X_cpsz, y_cpsz = self.generate_features(self.cpsz_stft_norm, 2)
        X_tcsz, y_tcsz = self.generate_features(self.tcsz_stft_norm, 3)


        # Append the matrices

        X = np.append(X_null, X_bckg, axis=0)
        X = np.append(X, X_gnsz, axis=0)
        X = np.append(X, X_cpsz, axis=0)
        X = np.append(X, X_tcsz, axis=0)

        y = np.append(y_null, y_bckg, axis=0)
        y = np.append(y, y_gnsz, axis=0)
        y = np.append(y, y_cpsz, axis=0)
        y = np.append(y, y_tcsz, axis=0)

        self.X = X
        self.y = y
        print("Training classifier with 5x5 CV...")
        self.scores_norm = self.classify(X, y, multiclass=True)

        return self.scores_norm


if __name__ == '__main__':

    dh = data_handling()
    dh.simulate()

    _, p_val = scipy.stats.ttest_ind(dh.scores, dh.scores_norm)
    print("Spectral Normalization improves classification accuracy by %0.2f percent on average; p_val = %0.2f" %((dh.scores_norm.mean() - dh.scores.mean())*100, p_val))
