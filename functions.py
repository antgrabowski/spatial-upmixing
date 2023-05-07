import numpy as np
import pysofaconventions as sofa
import matplotlib.pyplot as plt

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
        D = D * np.tile(scale ** -1, (F, 1))
        A = A * np.tile(scale ** 1, (N, 1)).T

        # Compute cost value
        cost[iter] = beta_divergence_cost(S, S_ap, beta)

    return D, A, cost

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
        cost = np.sum(S.flatten()**beta + (beta-1) * S_ap.flatten()**beta \
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
        wiener_filter = dictionary[:,i].reshape(-1,1) @ activation_matrix[i,:].reshape(1,-1)
        wiener_filter /= dynamic_range_limiting(modelled_power_spectrum, 120)

        # Compute the STFT of the filtered signal.
        half_stft = stft * wiener_filter
        full_stft = np.concatenate((half_stft, np.flipud(np.conj(half_stft[1:-1, :]))), axis=0)
        over_samples_separated_source = np.fft.ifft(full_stft, axis=0)

        # Take the real part of the signal
        separated_source_matrix = np.real(over_samples_separated_source[:segment_length, :])

        # Overlap-add the separated segments to obtain the final separated source.
        separated_sources[i,:] = overlap_add(separated_source_matrix, window, n_segments, n_overlap)

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
    for i in range(n_segments):
        out[idx] += window * matrix[:, i]
        idx += n_shift
    return out

def convolve_with_hrir(signal_front_left, signal_front_right, signal_back_left, signal_back_right, sofa_file_name):
    """
    Convolves the signals with the HRIRs.

    Args:
        signal_front_left (array): The signal from the front left loudspeaker.
        signal_front_right (array): The signal from the front right loudspeaker.
        signal_back_left (array): The signal from the back left loudspeaker.
        signal_back_right (array): The signal from the back right loudspeaker.
        sofa_file_name (string): The name of the SOFA file.

    Returns:
        left_channel (array): The left channel of the binaural signal.
        right_channel (array): The right channel of the binaural signal.
    """
    # Load the SOFA file
    mysofa = sofa.SOFAFile(sofa_file_name)

    # Get the impulse responses for the desired angles
    hrir_azi_front_left = mysofa.get_hrir(-30, 0, 0, 'left')
    hrir_azi_front_right = mysofa.get_hrir(30, 0, 0, 'right')
    hrir_azi_back_left = mysofa.get_hrir(-120, 0, 0, 'left')
    hrir_azi_back_right = mysofa.get_hrir(120, 0, 0, 'right')

    # Convolve the signals with the HRIRs
    convolved_front_left = np.convolve(signal_front_left, hrir_azi_front_left, mode='same')
    convolved_front_right = np.convolve(signal_front_right, hrir_azi_front_right, mode='same')
    convolved_back_left = np.convolve(signal_back_left, hrir_azi_back_left, mode='same')
    convolved_back_right = np.convolve(signal_back_right, hrir_azi_back_right, mode='same')

    # Normalize the output signals to avoid clipping
    convolved_front_left /= np.max(np.abs(convolved_front_left))*2
    convolved_front_right /= np.max(np.abs(convolved_front_right))*2
    convolved_back_left /= np.max(np.abs(convolved_back_left))*2
    convolved_back_right /= np.max(np.abs(convolved_back_right))*2

    right_channel = convolved_front_right + convolved_back_right
    left_channel = convolved_front_left + convolved_back_left

    return left_channel, right_channel
