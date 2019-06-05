import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from scipy.signal import butter, sosfiltfilt


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
    eeg_preproc = butter_bandpass_filter(eeg_preproc, lowcut, highcut, fs)

    return eeg_preproc