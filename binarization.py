import soundfile as sf
import sounddevice as sd
import numpy as np
import pysofaconventions as sofa

# Load the SOFA file
filename = 'my_hrtf.sofa'
hrtf = sofa.SOFAFile(filename)

# Load the audio files for each source
source_files = ['source1.wav', 'source2.wav', 'source3.wav']
sources = [sf.read(source)[0] for source in source_files]

# Set the sampling rate and block size
samplerate = hrtf.Data.SamplingRate
blocksize = 1024

# Initialize the output array
output = np.zeros((len(sources), len(hrtf.Data.IR)))

# Convolve each source with the HRTF
for i, source in enumerate(sources):
    for j in range(len(hrtf.Data.IR)):
        output[i,j] = np.convolve(source, hrtf.Data.IR[j,:], mode='same')

# Sum the output for all sources
mixed_output = np.sum(output, axis=0)

# Play the mixed output
sd.play(mixed_output, samplerate=samplerate)

# Wait for the playback to finish
sd.wait()