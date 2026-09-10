import numpy as np
from scipy.io import wavfile

# Synthesizes the demo input for main.py: two piano-like notes, first played apart, then together

fs = 48000  # matches the sampling rate of the HRIRs in b_nh68.sofa
duration = 3.5
t = np.arange(int(duration * fs)) / fs

def piano_note(f0, onset, n_partials=10, decay_time=0.5):
    """
    Synthesizes a piano-like note: harmonic partials with a fast attack and an exponential decay.

    Args:
        f0 (float): Fundamental frequency in Hz.
        onset (float): Onset time in seconds.
        n_partials (int): Number of partials.
        decay_time (float): Time constant of the exponential decay in seconds.

    Returns:
        note (array): The note, zero before its onset.
    """
    time_since_onset = t - onset
    partials = np.arange(1, n_partials + 1)
    note = np.sum(np.sin(2 * np.pi * f0 * np.outer(partials, time_since_onset)) / partials[:, None], axis=0)
    envelope = np.minimum(time_since_onset / 0.005, 1) * np.exp(-time_since_onset / decay_time)
    return np.where(time_since_onset >= 0, note * envelope, 0)

# C4 at 0 s, A4 at 1 s, both at 2 s
c4 = piano_note(261.63, 0.0) + piano_note(261.63, 2.0)
a4 = piano_note(440.0, 1.0) + piano_note(440.0, 2.0)

signal = 0.9 * (c4 + a4) / np.max(np.abs(c4 + a4))

# Saved as float32 without added noise: the IS-divergence NMF weighs every bin by its relative error, so a constant
# noise floor (including 16-bit quantization noise) pulls the two components away from the two notes
wavfile.write('twoPianoTones.wav', fs, signal.astype(np.float32))
