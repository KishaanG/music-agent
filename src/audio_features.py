import sys

import librosa
import bpm_finder
import pyloudnorm as pyln

import scipy.stats
from scipy.signal import find_peaks, butter, filtfilt
from scipy.interpolate import make_smoothing_spline

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

__all__ = [ "AudioFeatures" ]

# Fit tempo to a range of 70 to 180, as for example 60 bpm = 120 bpm or 200 bpm = 100 bpm
# This is the range that bpm_finder uses, which is used for bucketing the tempos
MINIMUM_TEMPO = 70
MAXIMUM_TEMPO = 180

# The number of samples between successive frames, set to 128 rather than 512 for higher accuracy.
HOP_LENGTH = 128

# Frequencies below this will be considered "Bass" in bass related functions
LOWPASS_FREQUENCY = 250

class AudioFeatures():
    def __init__(self, path):
        if path is None or not path.endswith('.wav'):
            raise FileNotFoundError(f'Invalid file "{path}" : Only .wav files are accepted.')
        
        self._path = path
        self._loadSong()
        self._getTempo()
        self._getLoudness()
        self._getMainFrequencies()
        self._getDynamics()
        self._getPeaks()
        self._getBeatStrength()

    ''' FEAUTURE OBJECT STRUCTURE '''
    # Only gets features and values used in model. There is extra data used to plot graphs
    def getFeatures(self):
        return {
            "tempo": self._static_tempo,
            "tempo_bucket": self._tempo_bucket,
            "beat_strength": self._beat_strength,
            "integrated_loudness": self._integrated_loudness,
            "loudness_range": self._loudness_range,
            "main_frequency": self._spectral_centroid_mean,
            "main_frequency_range": self._spectral_centroid_range,
            "time_to_peak": self._first_peak_time,
            "volume_mean": self._volume_mean,
            "volume_std": self._volume_std,
            "bass_volume_mean": self._bass_volume_mean,
            "bass_volume_std": self._bass_volume_std,
            "bass_difference": self._bass_difference,
        }

    def plot(self):
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 10), layout="constrained")
        self._getTempoPlot(fig=fig, ax=ax[0, 0])
        self._getMainFrequencyPlot(fig=fig, ax=ax[0, 1])
        self._getDynamicsPlot(ax=ax[1, 0])
        self._getBeatStrengthPlot(ax=ax[1, 1])
        plt.show()

    ''' GETTING FEATURE DATA'''
    def _loadSong(self):
        self._y, self._sr = librosa.load(self._path)
        self._onset_env = librosa.onset.onset_strength(y=self._y, sr=self._sr, hop_length=HOP_LENGTH)

    '''
    BUCKET     |BPM RANGE
    -----------|---------
    Slow       |70-89
    Mid-Tempo  |90-119
    Upbeat     |120-149
    Fast       |150-180
    '''
    def _getTempo(self):
        # Fit tempo to a range of 70 to 180, as for example 60 bpm = 120 bpm or 200 bpm = 100 bpm
        # This is the range that bpm_finder uses, which is used for bucketing the tempos
        prior = scipy.stats.uniform(loc=MINIMUM_TEMPO, scale=MAXIMUM_TEMPO-MINIMUM_TEMPO)
        
        self._dynamic_tempo = librosa.feature.tempo(onset_envelope=self._onset_env, sr=self._sr, aggregate=None, prior=prior, hop_length=HOP_LENGTH)    
        self._static_tempo = librosa.feature.tempo(onset_envelope=self._onset_env, sr=self._sr, prior=prior, hop_length=HOP_LENGTH).item()
        self._tempo_bucket = bpm_finder.tempo_bucket(self._static_tempo)

    def _getLoudness(self):
        meter = pyln.Meter(self._sr)
        self._integrated_loudness = meter.integrated_loudness(self._y).item()
        self._loudness_range = meter.loudness_range(self._y).item()

    def _getMainFrequencies(self):
        self._spectral_centroid = librosa.feature.spectral_centroid(y=self._y, sr=self._sr, hop_length=HOP_LENGTH)[0]
        
        times = librosa.times_like(self._spectral_centroid, hop_length=HOP_LENGTH)
        spline = make_smoothing_spline(times, self._spectral_centroid, lam=1e2)
        self._smooth_spectral_centroid = spline(times)

        self._spectral_centroid_mean = np.average(self._spectral_centroid).item()
        self._spectral_centroid_min = np.min(self._smooth_spectral_centroid).item()
        self._spectral_centroid_max = np.max(self._smooth_spectral_centroid).item()
        self._spectral_centroid_range = self._spectral_centroid_max - self._spectral_centroid_min

    def _getDynamics(self):
        self._rms = librosa.feature.rms(y=self._y, hop_length=HOP_LENGTH)[0]

        self._volume = librosa.amplitude_to_db(self._rms)
        self._volume_mean = self._volume.mean().item()
        self._volume_std = self._volume.std().item()

        # Create Low Pass filter at LOWPASS_FREQUENCY (i.e., only keep frequencies under LOWPASS_FREQUENCY)
        b, a = butter(N=4, Wn=LOWPASS_FREQUENCY, btype='lowpass', fs=self._sr)
        # Pass audio time-series through Low Pass Filter
        bass = filtfilt(b, a, self._y)

        self._bass_rms = librosa.feature.rms(y=bass, hop_length=HOP_LENGTH)
        self._bass_volume = librosa.amplitude_to_db(self._bass_rms)
        self._bass_volume_mean = self._bass_volume.mean().item()
        self._bass_volume_std = self._bass_volume.std().item()

        self._bass_difference = self._volume_mean - self._bass_volume_mean

    def _getPeaks(self):
        # HOW ARE PEAKS CHOSEN?
        # height: Minimum height to be considered a peak
        # distance: Minimum frame distance between two peaks
        # prominence: Minimum drop height after peak
        self._peaks, properties = find_peaks(
            x=self._rms,
            height=0.7 * self._rms.max(),       # 70% of the max volume in song
            distance=0.5 * self._sr / HOP_LENGTH,      # 0.5 seconds
            prominence=0.1 * self._rms.max()    # 10% of the max volume in song
        )

        times = librosa.times_like(self._rms, sr=self._sr, hop_length=HOP_LENGTH)

        self._peak_times = times[self._peaks]
        self._peak_heights = librosa.amplitude_to_db(properties["peak_heights"])
        self._first_peak_time = self._peak_times[0].item() 

    def _getBeatStrength(self):
        beat_differences = []
        _, self._beat_track = librosa.beat.beat_track(y=self._y, sr=self._sr, hop_length=HOP_LENGTH, start_bpm=self._static_tempo)
        beat_interval = np.median(np.diff(self._beat_track))

        # For every beat: 
        #   What is the difference in dB between the first third (hit) and last two-thirds (rest) of the beat?
        #   - For the hit, use the maximum volume to capture the loudest "hit" or "accentuation"
        #   - For the rest, use the mean volume to capture how the rest of the beat "sounds" after
        # This metric will not be 100% accurate as for example, for drums, different kicks have different transients (punch) lengths. 
        # We do not know how long the beat will be emphasized for. This is to attempt to separate the impact/smoothness coming from a string quartet compared to an electronic song with heavy kicks and basslines.
        third = int(round(beat_interval/3))
        for i in range(len(self._beat_track)-1):
            start = self._beat_track[i]
            end = self._beat_track[i+1]
            hit = self._volume[start:start + third]
            rest = self._volume[start + third:end]
            beat_differences.append(abs(hit.max() - rest.mean()))
        
        self._beat_strength = np.median(beat_differences).item()
        beat_differences.append(0)
        self._beat_differences = np.array(beat_differences)

    ''' CREATE GRAPHS '''
    def _format_min_sec(self):
        return FuncFormatter(lambda seconds, _: f"{int(seconds // 60)}:{int(seconds % 60):02d}")

    def _getTempoPlot(self, fig, ax):
        tempogram = librosa.feature.tempogram(onset_envelope=self._onset_env, sr=self._sr, hop_length=HOP_LENGTH)
        img = librosa.display.specshow(tempogram, sr=self._sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='tempo', cmap='magma', ax=ax)

        times = librosa.times_like(self._dynamic_tempo, sr=self._sr, hop_length=HOP_LENGTH)

        ax.plot(times, self._dynamic_tempo, linewidth=2, color='white', label='Tempo Estimate')
        ax.axhline(self._static_tempo, linewidth=2, color='cyan', linestyle='--', label=f'Estimated tempo (~{int(round(self._static_tempo))} bpm)')

        ax.set_ylim(MINIMUM_TEMPO, MAXIMUM_TEMPO)
        ax.set_yticks([70, 90, 120, 150])
        ax.set_yticklabels(["Slow (70)", "Mid-Tempo (90)", "Upbeat (120)", "Fast (150)"])

        ax.set_xlabel("Tempo (BPM)")
        ax.set_ylabel("Time (m:s)")
        ax.legend(loc='upper right')
        ax.set(title='Estimated Tempo over Time')

        fig.colorbar(img, ax=ax, label='Correlation')

    def _getMainFrequencyPlot(self, fig, ax):
        stft = librosa.stft(self._y, hop_length=HOP_LENGTH)
        times = librosa.times_like(self._spectral_centroid, hop_length=HOP_LENGTH)

        img = librosa.display.specshow(stft, sr=self._sr, vscale='dBFS', y_axis='log', x_axis='time', ax=ax, hop_length=HOP_LENGTH)

        ax.plot(times, self._spectral_centroid.T, label='Spectral Centroid (Raw)', color='black', alpha=0.5)
        ax.plot(times, self._smooth_spectral_centroid, label='Spectral Centroid (Smooth)', color='red')
        ax.axhline(y=self._spectral_centroid_mean, label='Mean Frequency', color='cyan', linestyle='-')
        ax.axhline(y=self._spectral_centroid_min, label='Mininum Frequency', color='green', linestyle='--')
        ax.axhline(y=self._spectral_centroid_max, label='Maximum Frequency', color='yellow', linestyle='--')

        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (m:s)")
        ax.set_title("Frequency over Time")
        ax.legend(loc='lower right')

        fig.colorbar(img, ax=ax, label='Volume (dB)')

    def _getDynamicsPlot(self, ax):
        times = librosa.times_like(self._rms, sr=self._sr, hop_length=HOP_LENGTH)

        ax.plot(times, self._volume, label="d=Volume", alpha=0.25)
        ax.plot(times, self._bass_volume.T, label="Bass Volume", alpha=0.25)

        ax.axhline(self._volume_mean, color="red", label=f"Mean (~{self._volume_mean:.2f} dB)")
        ax.axhline(self._bass_volume_mean, color="blue", label=f"Bass Mean (~{self._bass_volume_mean:.2f} dB)")

        min_freq = self._volume_mean - self._volume_std
        max_freq = self._volume_mean + self._volume_std
        ax.axhspan(min_freq, max_freq, color="yellow", alpha=0.25, label=f"+/-1 std(~{self._volume_std:.2f}) dB")

        ax.axhspan(self._bass_volume_mean, self._volume_mean, color="black", alpha=0.25, label=f"Bass Volume Difference: (~{self._bass_difference:.2f} dB)")
        
        ax.axvline(self._peak_times[0], color="green", linestyle="--", label="First Peak", alpha=0.5)
        ax.vlines(self._peak_times, ymin=self._peak_heights, ymax=self._peak_heights-0.1, color="black", linestyle=':', label="Peaks")
        
        ax.set_ylabel("Volume (dB)")
        ax.set_title(f'Volume over Time')
        ax.xaxis.set_major_formatter(self._format_min_sec())
        ax.set_xlabel("Time (m:s)")
        ax.legend(loc="lower right")
        ax.label_outer()

    def _getBeatStrengthPlot(self, ax):
        times = librosa.times_like(self._onset_env, sr=self._sr, hop_length=HOP_LENGTH)

        ax.plot(times[self._beat_track], self._beat_differences, color="blue", label="Beat Difference")
        ax.axhline(self._beat_strength, color="red", label=f"Mean Beat Difference (~{self._beat_strength:.2f})")
        ax.vlines(times[self._beat_track][::4], 0, self._beat_differences.max(), color="black", label="Bars", linestyle="-", alpha=0.2)

        ax.set_title("Beat Strength (i.e., Beat (hit - rest))")
        ax.set_ylabel("Change in Volume (dB)")
        ax.set_xlabel("Time (m:s)")
        ax.xaxis.set_major_formatter(self._format_min_sec())
        ax.legend(loc="lower right")

def main():
    a = AudioFeatures(sys.argv[1])
    print(a.getFeatures())
    a.plot()

if __name__ == "__main__":
    main()