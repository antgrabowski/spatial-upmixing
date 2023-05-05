import librosa 
import numpy as np
import pysofaconventions as sofa


# NMF ambient sound separation
# Takes in a stereo audio file and separates the sources using the guided NMF algorithm.
def nmf_ambient_sound_separation(audio_file):
    """
    Separates the sources from a stereo audio file using the guided NMF algorithm.

    Args:
        audio_file (string): Path to the audio file.

    Returns:
        audio_out (array): Separated audio signals.
    """
    # load the audio file
    audio, fs = librosa.load(audio_file, fs=None, mono=False)

    # apply the guided NMF algorithm to the left channel
    W1, H1 = guided_nmf(audio[0, :], 3, 200, 1024, 512, fs)

    # apply the guided NMF algorithm to the right channel
    W2, H2 = guided_nmf(audio[1, :], 3, 200, 1024, 512, fs)

    # apply the SOFA filters
    H1 = apply_sofa_filter(H1, 'data/sofa/HRIRs_CIRCULAR_9.5cm_radius.sofa', 0, 0)
    H2 = apply_sofa_filter(H2, 'data/sofa/HRIRs_CIRCULAR_9.5cm_radius.sofa', 0, 1)

    # apply the Wiener filter to the activation vectors
    H1 = wiener_filter(H1, X, 200, 1024, 512, fs)
    H2 = wiener_filter(H2, X, 200, 1024, 512, fs)

    # apply the Wiener filter to the activation vectors
    H1 = wiener_filter(H1, X, 200, 1024, 512, fs)
    H2 = wiener_filter(H2, X, 200, 1024, 512, fs)

    # apply the SOFA filters
    H1 = apply_sofa_filter(H1, 'data/sofa/HRIRs_CIRCULAR_9.5cm_radius.sofa', 0, 0)
    H2 = apply_sofa_filter(H2, 'data/sofa/HRIRs_CIRCULAR_9.5cm_radius.sofa', 0, 1)

    # apply the Wiener filter to the activation vectors
    H1 = wiener_filter(H1, X, 200, 1024, 512, fs)
    H2 = wiener_filter


# Guided NMF algorithm for source separation
def guided_nmf(audio, n_components, n_iter, n_fft, hop_length, beta):
    """
    Applies the guided NMF algorithm to separate the sources from an audio mixture.
    implement NMF yourself using the multiplicative update rule

    Args:
        audio (array): Audio mixture.
        n_components (int): Number of components.
        n_iter (int): Number of iterations.
        n_fft (int): FFT size.
        hop_length (int): Hop length.
        beta (int): Beta divergence value.

    Returns:
        D (array): Dictionary 
        A (array): Activation 
    """
    # get the magnitude spectrogram
    X = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    X = np.abs(X)

    # initialize the dictionary and activation matrices
    D = np.random.rand(X.shape[0], n_components)
    A = np.random.rand(n_components, X.shape[1])

    # iterate
    for i in range(n_iter):
        # update the activation matrix
        A = A * (np.dot(D.T, X) / (np.dot(np.dot(D.T, D), A) + 1e-9))

        # update the dictionary matrix
        D = D * (np.dot(X, A.T) / (np.dot(np.dot(D, A), A.T) + 1e-9))

    return D, A
    

def apply_sofa_filter(audio, sofa_file, source, receiver):
    """
    Applies a SOFA filter to an audio signal.

    Args:
        audio (array): Audio signal.
        sofa_file (string): Path to the SOFA file.
        source (int): Source index.
        receiver (int): Receiver index.

    Returns:
        audio_out (array): Filtered audio signal.
    """
    # load the SOFA file
    sofa_file = sofa.SOFAFile(sofa_file)

    # get the filter coefficients
    ir = sofa_file.Data.IR.get_values()
    ir = ir[source, receiver, :]

    # apply the filter
    audio_out = np.convolve(audio, ir)

    return audio_out

# wiener filter taking the NMF output as input and returning the separated sources
def wiener_filter(audio, n_components, n_fft, hop_length, fs):
    """
    Applies a Wiener filter to an audio signal.

    Args:
        audio (array): Audio signal.
        n_components (int): Number of components.
        n_fft (int): FFT size.
        hop_length (int): Hop length.
        fs (int): Sampling frequency.

    Returns:
        audio_out (array): Filtered audio signal.
    """
    # get the magnitude spectrogram
    X = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    X = np.abs(X)

    # get the power spectrogram
    P = np.power(X, 2)

    # get the noise power
    Pn = np.mean(P[:, 0:100], axis=1)

    # get the SNR
    SNR = P / Pn[:, None]

    # apply the Wiener filter
    G = np.power(SNR / (1 + SNR), 2)

    # apply the filter
    audio_out = np.multiply(G, X)

    # convert the filtered signal back to the time domain
    audio_out = librosa.istft(audio_out, hop_length=hop_length)

    return audio_out
