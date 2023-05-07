import functions as f
import librosa
import numpy as np
from scipy.signal import hann, stft

# load an audio file 
audio, fs = librosa.load('twoPianoTones.wav', sr=None)

# parameters
segment_time = 0.05
segment_length = 2 * round(segment_time * fs / 2)  # ensure an even segment length
window = np.sqrt(hann(segment_length, sym=False))
n_overlap = segment_length // 2
n_fft = 2 * 4096
n_sources = 2
sofa_name = 'b_nh68.sofa'

freq_vector, time_vector, stft_out = stft(audio, fs=fs, window=window, nperseg=segment_length, noverlap=n_overlap, nfft=n_fft)
power_spectrum = np.abs(stft_out) ** 2 / len(window)

# initialize the dictionary matrix
D_init = np.random.rand(power_spectrum.shape[0], n_sources)

# initialize the activation matrix
A_init = np.random.rand(n_sources, power_spectrum.shape[1])

# run the beta-NMF algorithm
D, A, cost = f.beta_nmf_mu(power_spectrum, 100, D_init, A_init, 0)
# f.plot_data(A[0], fs, 'Activation matrix', 'Time (s)', 'Amplitude')

# separate the sources using the Wiener filter
separated_sources = f.wiener_nmf_separation(stft_out, D, A, window, n_overlap)

# normalize the separated sources
max_value = np.max(np.abs(separated_sources))
for i in range(n_sources):
    separated_sources[i] = separated_sources[i] / (max_value * 0.9)

# binarize the separated sources
#f.convolve_with_hrir(separated_sources_left[0], separated_sources_right[0], separated_sources_left[1], separated_sources_left[1], 'sofa_name')

for i in range(n_sources):
    f.plot_data(separated_sources[i], fs, 'Separated source ' + str(i+1) , 'Time (s)', 'Amplitude')

# print(cost)