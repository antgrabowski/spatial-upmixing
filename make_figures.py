import os
import runpy
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import spectrogram

# Renders the README figures from a run of main.py (run generate_signal.py first)

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
AXIS = '#c3c2b7'
# One-hue sequential ramp (light -> dark blue); the lowest levels fade into the background
LEVEL_CMAP = LinearSegmentedColormap.from_list('level', [SURFACE, '#cde2fb', '#9ec5f4', '#6da7ec',
                                                         '#3987e5', '#256abf', '#184f95', '#0d366b'])

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
    'axes.titlecolor': INK, 'axes.titlesize': 11, 'axes.titlelocation': 'left',
    'axes.labelcolor': INK_SECONDARY, 'axes.labelsize': 9,
    'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': INK_SECONDARY, 'ytick.labelcolor': INK_SECONDARY,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
})

def plot_spectrograms(signals, titles, fs, file_name, dynamic_range_db=60, max_freq_khz=5):
    """
    Plots stacked spectrograms on one dB scale (0 dB = loudest bin in the figure) and saves them to figures/.

    Args:
        signals (list of arrays): Signals to plot, one panel each.
        titles (list of strings): Panel titles.
        fs (int): Sampling frequency.
        file_name (string): Output file name.
        dynamic_range_db (float): Range of the colour scale in dB.
        max_freq_khz (float): Upper limit of the frequency axis in kHz.
    """
    levels = []
    for signal in signals:
        freqs, times, power = spectrogram(signal, fs=fs, window='hann', nperseg=2048, noverlap=2048 - 256)
        levels.append((times, freqs, 10 * np.log10(power + 1e-20)))
    peak = max(level.max() for _, _, level in levels)

    fig, axes = plt.subplots(len(signals), 1, figsize=(8, 0.6 + 2.2 * len(signals)), sharex=True, squeeze=False,
                             layout='constrained')
    axes = axes[:, 0]
    for ax, (times, freqs, level), title in zip(axes, levels, titles):
        mesh = ax.pcolormesh(times, freqs / 1000, level - peak, cmap=LEVEL_CMAP, vmin=-dynamic_range_db, vmax=0,
                             shading='auto')
        ax.set_ylim(0, max_freq_khz)
        ax.set_title(title)
        ax.set_ylabel('Frequency (kHz)')
    axes[-1].set_xlabel('Time (s)')

    colorbar = fig.colorbar(mesh, ax=axes, label='Level (dB re. peak)')
    colorbar.outline.set_edgecolor(AXIS)
    colorbar.outline.set_linewidth(0.8)

    fig.savefig(os.path.join('figures', file_name), dpi=150)
    plt.close(fig)

np.random.seed(0)  # main.py initializes the NMF with np.random.rand; seed it so the figures are reproducible
warnings.filterwarnings('ignore', message='FigureCanvasAgg is non-interactive')  # main.py calls plt.show()
sys.argv = ['main.py', 'twoPianoTones.wav']  # main.py's input argument: the demo signal from generate_signal.py
results = runpy.run_path('main.py')
plt.close('all')

os.makedirs('figures', exist_ok=True)
fs = results['fs']
plot_spectrograms([results['audio']], ['Input mixture'], fs, 'input.png')
plot_spectrograms(results['separated_sources'], ['Source 1', 'Source 2'], fs, 'separation.png')
plot_spectrograms([results['binaural_left'], results['binaural_right']], ['Left ear', 'Right ear'], fs, 'binaural.png')
