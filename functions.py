import librosa 
import numpy as np
import pysofaconventions as sofa
import matplotlib.pyplot as plt
    
    
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

def beta_nmf_mu(S, n_iter, D, A, beta):
    """
    Applies the beta-NMF algorithm to separate the sources from an audio mixture.
    implement NMF yourself using the multiplicative update rule

    Args:
        S (array): Spectrogram of the mixture.
        n_iter (int): Number of iterations.
        D (array): Dictionary matrix.
        A (array): Activation matrix.
        beta (int): Beta divergence value.

    Returns:
        D (array): Dictionary
        A (array): Activation
        cost (array): Cost function
    """
    # Initialize variables
    F, N = S.shape
    cost = np.zeros(n_iter)

    # Compute data approximate
    S_ap = D @ A

    # Compute initial cost value
    cost[0] = beta_divergence_cost(S, S_ap, beta)

    for iter in range(1, n_iter):

        # Update W
        D = D * ((S * S_ap ** (beta - 2)) @ A.T) / (S_ap ** (beta - 1) @ A.T)
        S_ap = D @ A

        # Update H
        A = A * (D.T @ (S * S_ap ** (beta - 2))) / (D.T @ S_ap ** (beta - 1))
        S_ap = D @ A

        # Norm-2 normalization
        scale = np.sqrt(np.sum(D ** 2, axis=0))
        D = np.multiply(D, np.tile(scale ** -1, (F, 1)))
        A = np.multiply(A, np.tile(scale.reshape(-1, 1), (1, N)))

        # Compute cost value
        cost[iter] = beta_divergence_cost(S, S_ap, beta)

    return D, A, cost

# def wiener_nmf_separation(stft, dictionary, activation_matrix, window, n_overlap):
    """
    Separates the sources from a stereo audio file using the guided NMF algorithm.

    Args:
        stft (array): STFT of the mixture.
        dictionary (array): Dictionary matrix.
        activation_matrix (array): Activation matrix.
        window (array): Window function.
        n_overlap (int): Number of overlapping samples.

    Returns:
        separated_sources (array): Separated sources.
    """
    # get the dimensions
    segment_length = len(window)
    n_sources, n_segments = activation_matrix.shape
    n_shift = segment_length - n_overlap

    separated_sources = np.zeros((n_sources, n_segments*n_shift+n_overlap))
    modelled_power_spectrum = np.dot(dictionary, activation_matrix)

    # iterate over the sources
    for ii in range(n_sources):

        half_stft = stft*(dictionary[:,ii]*activation_matrix[ii,:])/dynamic_range_limiting(modelled_power_spectrum, 120)
        full_stft = np.concatenate((half_stft, np.flipud(np.conj(half_stft[1:-1,:]))), axis=0)

        over_samples_separated_source = np.fft.ifft(full_stft, axis=0)
        separated_source_matrix = np.real(over_samples_separated_source[0:segment_length,:])
        idx = np.arange(segment_length)

        # overlap-add the segments
        for jj in range(n_segments):
            separated_sources[ii,idx] = separated_sources[ii,idx] + (window*separated_source_matrix[:,jj]).T
            idx = idx + n_shift

    return separated_sources

def beta_divergence_cost(S, S_ap, beta):
    """
    Computes the beta divergence cost between two matrices.

    Args:
        S (array): Source matrix.
        S_ap (array): Approximation matrix.
        beta (int): Beta divergence value.

    Returns:
        cost (float): Beta divergence cost.
    """
    F, N = S.shape
    if beta == 0:
        # IS divergence
        cost = np.sum(S/S_ap - np.log(S/S_ap)) - F*N
    elif beta == 1:
        # KL divergence
        cost = np.sum(S*np.log(S/S_ap) + S - S_ap)
    else:
        # general beta divergence
        cost = np.sum(S.flatten()**beta + (beta-1) * S_ap.flatten()**beta /
                     - beta*S.flatten() * S_ap.flatten()**(beta-1) / (beta**2 - beta))
    return cost

def dynamic_range_limiting(nonnegative_data, max_range_db):
    """
    Applies dynamic range limiting to a nonnegative data vector.

    Args:
        nonnegative_data (array): Nonnegative data vector.
        max_range_db (float): Maximum dynamic range in dB.

    Returns:
        limited_nonnegative_data (array): Limited nonnegative data vector.
    """

    log_power_spectrum = 10 * np.log10(nonnegative_data)
    limited_log_power_spectrum = np.maximum(log_power_spectrum, 
                                             np.max(log_power_spectrum) - max_range_db)
    limited_nonnegative_data = 10 ** (limited_log_power_spectrum / 10)
    return limited_nonnegative_data

# plot data
def plot_data(data, fs, title, xlabel, ylabel, xlim=None, ylim=None):
    """
    Plots a data vector.

    Args:
        data (array): Data vector.
        fs (int): Sampling frequency.
        title (string): Title of the plot.
        xlabel (string): Label of the x-axis.
        ylabel (string): Label of the y-axis.
        xlim (array): Limits of the x-axis.
        ylim (array): Limits of the y-axis.
    """
    # get the time vector
    t = np.arange(0, len(data)/fs, 1/fs)

    # plot the data
    plt.figure()
    plt.plot(t, data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.grid()
    plt.show()

def wiener_nmf_separation(stft, dictionary, activation_matrix, window, n_overlap):
    """
    Separates the sources from the input mixture using Wiener-NMF algorithm.

    Args:
        stft (ndarray): The input mixture's short-time Fourier transform (STFT) matrix.
        dictionary (ndarray): The dictionary matrix learned by the NMF algorithm.
        activation_matrix (ndarray): The activation matrix learned by the NMF algorithm.
        window (ndarray): The window function to be applied to each segment during the inverse STFT.
        n_overlap (int): The number of samples of overlap between adjacent STFT segments.

    Returns:
        ndarray: The matrix containing the separated sources. Its shape is (n_sources, n_samples).

    """
    segment_length = len(window)
    n_sources, n_segments = activation_matrix.shape
    n_shift = segment_length - n_overlap
    separated_sources = np.zeros((n_sources, n_segments*n_shift+n_overlap), dtype=np.complex128)

    # Compute the power spectrum of the learned model.
    modelled_power_spectrum = np.dot(dictionary, activation_matrix)

    # Iterate over each source and separate it from the input mixture.
    for i in range(n_sources):
        # Compute the Wiener filter for the current source.
        wiener_filter = dictionary[:, i].reshape((-1, 1)) * activation_matrix[i, :]
        wiener_filter /= dynamic_range_limiting(modelled_power_spectrum, 120)

        # Compute the STFT of the filtered signal.
        half_stft = stft * wiener_filter
        full_stft = np.concatenate((half_stft, np.flipud(np.conj(half_stft[1:-1, :]))), axis=0)
        over_samples_separated_source = np.fft.ifft(full_stft, axis=0)

        # Take the real part of the signal to remove numerical errors.
        separated_source_matrix = np.real(over_samples_separated_source[:segment_length, :])

        # Overlap-add the separated segments to obtain the final separated source.
        idx = np.arange(segment_length)
        for j in range(n_segments):
            separated_sources[i, idx] += window * separated_source_matrix[:, j]
            idx += n_shift

        overlap_add(separated_source_matrix, window, n_segments, n_overlap)

    return separated_sources

def overlap_add(matrix, window, n_segments, n_overlap):
    """
    Overlap-adds the separated segments to obtain the final separated source.

    Args:
        matrix (ndarray): The matrix containing the separated segments.
        window (ndarray): The window function to be applied to each segment during the inverse STFT.
        n_overlap (int): The number of samples of overlap between adjacent STFT segments.

    Returns:
        ndarray: The matrix containing the separated sources. Its shape is (n_sources, n_samples).

    """
    segment_length = len(window)
    n_shift = segment_length - n_overlap

    out = np.zeros((n_segments*n_shift+n_overlap), dtype=np.complex128)
    idx = np.arange(segment_length)
    for j in range(n_segments):
        out[idx] += window * matrix[:, j]
        idx += n_shift
    return out

# plot spectrogram data
def plot_spectrogram(data, fs, title, xlabel, ylabel, xlim=None, ylim=None):
    """
    Plots the spectrogram of the input data.

    Args:
        data (ndarray): Input data.
        fs (int): Sampling frequency.
        title (string): Title of the plot.
        xlabel (string): Label of the x-axis.
        ylabel (string): Label of the y-axis.
        xlim (array): Limits of the x-axis.
        ylim (array): Limits of the y-axis.
    """
    # plot the spectrogram
    plt.figure()
    plt.specgram(data, Fs=fs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.colorbar()
    plt.show()