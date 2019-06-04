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

