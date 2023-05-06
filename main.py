import functions as f
import librosa
import numpy as np
from scipy.signal import hann, stft

# load an audio file 
audio, fs = librosa.load('twoPianoTones.wav', sr=None)

# ploting the audio signal
# f.plot_data(audio, fs, 'Audio data', 'Time (s)', 'Amplitude')


segment_time = 0.05
segment_length = 2 * round(segment_time * fs / 2)  # ensure an even segment length
window = np.sqrt(hann(segment_length, sym=False))
n_overlap = segment_length // 2
n_fft = 2 * 4096

freq_vector, time_vector, stft_out = stft(audio, fs=fs, window=window, nperseg=segment_length, noverlap=n_overlap, nfft=n_fft)
power_spectrum = np.abs(stft_out) ** 2 / len(window)


# initialize the dictionary matrix
D_init = np.random.rand(power_spectrum.shape[0], 1)

# initialize the activation matrix
A_init = np.random.rand(1, power_spectrum.shape[1])

# run the beta-NMF algorithm
D, A, cost = f.beta_nmf_mu(power_spectrum, 100, D_init, A_init, 2)
# f.plot_data(D[1], fs, 'Activation matrix', 'Time (s)', 'Amplitude')

freq_range = [0, 4000]
dyn_range_db = 60
#f.nmf_equation_plot(D, A, fs, segment_length, n_overlap, power_spectrum, freq_range, dyn_range_db)

# separate the sources using the Wiener filter
separated_sources = f.wiener_nmf_separation(stft_out, D, A, window, n_overlap)
# f.plot_data(separated_sources[0], fs, 'Separated source 1', 'Time (s)', 'Amplitude')

# calculate the difference of lengrh between the separated sources and the original audio and cut the sources
separated_sources = separated_sources[:, :audio.shape[0]]
f.plot_data(separated_sources[0], fs, 'Separated source 1', 'Time (s)', 'Amplitude')
# calculate a delay between the separated sources and the original audio using cross-correlation
delay = np.argmax(np.correlate(separated_sources[0], audio)) - audio.shape[0]
print(delay)

deviated_data = audio - separated_sources
f.plot_data(deviated_data[0], fs, 'Deviation', 'Time (s)', 'Amplitude')
# print(cost)