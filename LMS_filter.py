import numpy as np

def lms_filter(x, d, mu, M):
    """
    Applies the LMS algorithm to estimate the filter coefficients
    based on the input signal x and the desired signal d.

    Args:
        x (array): Input signal.
        d (array): Desired signal.
        mu (float): Step size parameter.
        M (int): Filter length.

    Returns:
        w (array): Estimated filter coefficients.
        y (array): Filter output.
    """
    N = len(x)
    w = np.zeros(M)
    y = np.zeros(N)

    for n in range(M-1, N):
        # Extract a segment of length M from the input signal
        x_seg = x[n-M+1:n+1]

        # Compute the filter output
        y[n] = np.dot(w, x_seg)

        # Compute the error
        e = d[n] - y[n]

        # Update the filter coefficients
        w = w + mu * e * x_seg

    return w, y