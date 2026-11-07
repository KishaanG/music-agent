import librosa
import bpm_finder
import pyloudnorm as pyln

import scipy.stats
import numpy as np

import matplotlib.pyplot as plt

# Fit tempo to a range of 70 to 180, as for example 60 bpm = 120 bpm or 200 bpm = 100 bpm
# This is the range that bpm_finder uses, which is used for bucketing the tempos
MINIMUM_TEMPO = 70
MAXIMUM_TEMPO = 180

# The number of samples between successive frames, set to 128 rather than 512 for higher accuracy.
HOP = 128

def getTempo(onset_env, sr):
    # Fit tempo to a range of 70 to 180, as for example 60 bpm = 120 bpm or 200 bpm = 100 bpm
    # This is the range that bpm_finder uses, which is used for bucketing the tempos
    prior = scipy.stats.uniform(loc=MINIMUM_TEMPO, scale=MAXIMUM_TEMPO-MINIMUM_TEMPO)
    
    dtempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=None, prior=prior, hop_length=HOP)    
    stempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, prior=prior, hop_length=HOP).item()

    '''
    BUCKET     |BPM RANGE
    -----------|---------
    Slow       |70-89
    Mid-Tempo  |90-119
    Upbeat     |120-149
    Fast       |150-180
    '''
    bucket = bpm_finder.tempo_bucket(stempo)

    tempo_data = {
        "static_tempo": stempo,
        "dynamic_tempo": dtempo,
        "tempo_bucket": bucket
    }

    return tempo_data

def displayTempogram(onset_env, sr, stempo, dtempo):
    _, ax = plt.subplots()
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=HOP)
    librosa.display.specshow(tempogram, sr=sr, hop_length=HOP, x_axis='time', y_axis='tempo', cmap='magma', ax=ax)

    hl = librosa.display.highlight(ax=ax, linewidth=3, color='k', alpha=0.75)
    times = librosa.times_like(dtempo, sr=sr, hop_length=HOP)

    ax.axhline(stempo, path_effects=hl, linewidth=2, color='k', linestyle='--', label='Estimated tempo {:g}'.format(stempo))
    ax.plot(times, dtempo, linewidth=2, path_effects=hl, label='Tempo Estimate')

    ax.set_ylim(MINIMUM_TEMPO, MAXIMUM_TEMPO)
    ax.set_yticks([70, 90, 120, 150])
    ax.set_yticklabels(["Slow (70)", "Mid-Tempo (90)", "Upbeat (120)", "Fast (150)"])

    ax.legend(loc='upper right')
    ax.set(title='Estimated Tempo over Time')

    plt.show()

def getLoudness(y, sr):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(y).item()
    loudness_range = meter.loudness_range(y).item()
    loudness_data = {
        "integrated": loudness,
        "range": loudness_range
    }
    return loudness_data

'''MAIN FEATURE FUNC'''
def extractFeatures(path):
    y, sr = librosa.load(path)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)

    tempo_data = getTempo(onset_env=onset_env, sr=sr)
    
    loudness_data = getLoudness(y, sr)

    song_data = {
        "onset_env": onset_env,
        "sr": sr,
        "tempo": tempo_data,
        "loudness": loudness_data
    }

    print(song_data)

filepath = './test.wav'
extractFeatures(filepath)