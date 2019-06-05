import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import confusion_matrix
from seaborn import heatmap

def load_text(PATH=''):
    words = []
    onsets = []
    offsets = []
    sentences = []

    for jj in range(1,21):
        text_temp = loadmat(f'{PATH}Stimuli/Text/Run{jj}.mat')
        words.append(text_temp['wordVec'].ravel())
        onsets.append(text_temp['onset_time'].ravel())
        offsets.append(text_temp['offset_time'].ravel())
        sentences.append(text_temp['sentence_boundaries'].ravel())

    return words, onsets, offsets, sentences


def load_one_subject(subject, PATH=''):
    EEG = []
    mastoids = []
    for jj in range(1, 21):
        eeg_temp = loadmat(f'{PATH}EEG/{subject}/{subject}_Run{jj}.mat')
        EEG.append(eeg_temp['eegData'])
        mastoids.append(eeg_temp['mastoids'])

    return EEG, mastoids


def words_to_onsets(onsets, words, word_counts, word_keys,
                    wc_threshold=30, fs=128):

    words_and_samples = {}
    # Onsets from seconds to samples
    onsets_as_samples = [(i * fs).astype(int) for i in onsets]
    target_words = word_keys[word_counts > wc_threshold]
    # Create dictionary with word onsets
    for tw in target_words:
        words_and_samples[tw] = []
        for w, oas in zip(words, onsets_as_samples):
            words_and_samples[tw].append(oas[w==tw])

    return words_and_samples


def word_histogram(words):
    # Create a histogram of words
    word_list = [i.item() for i in np.hstack(words)]
    word_dict = {}
    for i in word_list:
        if i in word_dict.keys():
            word_dict[i] += 1
        else:
            word_dict[i] = 1

    # Transforming counts and keys to lists
    word_keys = list(word_dict.keys())
    word_counts = list(word_dict.values())
    sort_idx = np.argsort(word_counts)[::-1]
    word_keys = np.array(word_keys)[sort_idx]
    word_counts = np.array(word_counts)[sort_idx]

    return word_keys, word_counts


def word_dict_summary(word_samples_ons, word_samples_ofs, summary_func):
    '''
    word_samples_ons output of words_to_onset (with onsets)
    word_samples_ofs output of words_to_onset (with ofset)
    summary_func: a function to calculate summaries (i.e. np.max)
    return the summary across all words
    '''
    descript = []
    for word in word_samples_ons.keys():
        w_descript = []
        tmp_on = word_samples_ons[word]
        tmp_of = word_samples_ofs[word]
        for (tn, tf) in zip(tmp_on, tmp_of):
            if len(tn) != 0:
                w_descript.append(summary_func(tf - tn))
            else:
                w_descript.append(0)

        descript.append(summary_func(w_descript))

    return descript


def butter_bandpass(lowcut, highcut, fs, order=5):
    '''
    lowcut for bandpass
    highcut for bandpass
    fs frequency of signal
    order the filter order.
    Source: https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    and changed following:
    https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band',
                 analog=False, output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)
    return y


def simple_EEG_preprocessing(data, lowcut, highcut, fs):
    '''
    data = n_timepoints x n_channels
    lowcut = lower cutoff for bandpass
    highcut = higher cutoff for bandpass
    fs = fs of the EEG signal
    '''
    LR = LinearRegression() # Using intercept should already remove the
    # channel mean. Linear regression for linear detrending.
    eeg_preproc = data - np.mean(data, 1, keepdims=True) # average reference
    LR.fit(np.arange(data.shape[0]).reshape(-1,1), eeg_preproc) # fit LR
    eeg_preproc = (eeg_preproc -
                   LR.predict(np.arange(data.shape[0]).reshape(-1,1)))
    # Remove linear trend, for each channel.
    # Bandpass filter
    eeg_preproc = butter_bandpass_filter(eeg_preproc.T, lowcut, highcut, fs).T

    return eeg_preproc


def lag_builder(time_min, time_max):
    """
    Copied from unpublished pymtrf toolbox. Release (hopefully soon)
    Build the lags for the lag_generator function. Basically the indices of
    the time lags (including the starting and stopping points) of the data
    matrix.

    Parameters
    ----------
    time_min : np.int
        The starting index of the matrix as integer.
    time_max : np.int
        The stopping index of the matrix as integer.

    Returns
    -------
    lag_vector : numpy.ndarray, shape (np.abs(time_max) + np.abs(time_min) + 1,)
        A numpy array including all the lags.
    """

    if time_min > time_max:
        lag_vector = np.arange(time_max, time_min + 1)[::-1]
    else:
        lag_vector = np.arange(time_min, time_max + 1)

    return lag_vector


def lag_gen(data, time_lags):
    '''
    Not yet published pymtrf toolbox
    lag_gen returns the matrix containing the lagged time series of data for
    a range of time lags given by the list or numpy array lags. If the data is
    multivariate, lag_gen concatenates the features for each lag along the
    columns of the output array.

    Parameters
    ----------
    data : {float, array_like}, shape = [n_samples, n_features]
        The training data, i.e. the data that is shifted in time.
    time_lags : {int, array_like}, shape = [n_lags]
        Indices for lags that will be applied to the data.

    Returns
    -------
    lagged_data : {float, array_like}, shape = [n_samples, n_features * n_lag}
        The data shifted in time, as described above.

    See also
    --------
    mtrf_train : calculate forward or backward models.
    mtrf_predict : predict stimulus or response based on models.
    mtrf_crossval : calculate reconstruction accuracies for a dataset.

    Translation to Python: Simon Richard Steinkamp
    Github:
    October 2018; Last revision: 18.January 2019
    Original MATLAB toolbox, mTRF v. 1.5
    Author: Michael Crosse
    Lalor Lab, Trinity College Dublin, IRELAND
    Email: edmundlalor@gmail.com
    Website: http://lalorlab.net/
    April 2014; Last revision: 18 August 2015
    '''
    lagged_data = np.zeros((data.shape[0], data.shape[1] * time_lags.shape[0]))

    chan = 0
    for lags in time_lags:

        if lags < 0:
            lagged_data[:lags, chan:chan + data.shape[1]] = data[-lags:, :]
        elif lags > 0:
            lagged_data[lags:, chan:chan + data.shape[1]] = data[:-lags, :]
        else:
            lagged_data[:, chan:chan + data.shape[1]] = data

        chan = chan + data.shape[1]

    return lagged_data


def get_data_for_word(data, word_sample, idx, time_win=86):
    '''
    uses the the word sample dictionary to extract labels and samples.
    data: eeg in form n_timepoints x  n_channels, for one run
    Word sample:  contains for each data set the labels and the onsets.
    idx: the run which is represented by a run.
    time_win: how many timepoints after each onset are included

    '''
    data_sample = []
    labels = []
    for word in word_sample.keys():
        for on in word_sample[word][idx]:
            data_sample.append(data[on : on + time_win, :])
            labels.append(word)

    return data_sample, labels


def plot_confusion_matrix(y, y_pred, labels, normalize=True, cmap='coolwarm'):
    '''
    Plots a confusion matrix. Normalized or raw values.
    '''
    cm = confusion_matrix(y, y_pred)

    if normalize:
        cm = cm / np.sum(cm, 1, keepdims=True)

    heatmap(cm, annot=True, xticklabels= np.unique(labels),
            yticklabels=np.unique(labels), cmap='coolwarm')

    return None


def wrap_cross_val_predict(clf, X, y, cv):
    '''
    clf - An sklearn estimator
    X - data, n_features x n_samples
    y - labels for classification
    cv - a crossvalidation object (i.e. Kfold)
    Returns:
    trues - the correct values
    predictions - predictions
    '''

    predictions = []
    trues = []
    for tr, te in cv.split(X, y):
        clf.fit(X[tr], y[tr])
        predictions.append(clf.predict(X[te]))
        trues.append(y[te])

    return trues, predictions


