import numpy as np
import pandas as pd
from scipy.io import loadmat


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