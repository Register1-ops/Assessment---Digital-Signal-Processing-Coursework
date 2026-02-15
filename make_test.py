import numpy as np
from scipy.io.wavfile import write

# Parameters
fs = 44100        # sample rate
duration = 3      # seconds
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Make a simple tone (440 Hz sine) + a bit of noise
signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))

# Normalize to [-1, 1]
signal = signal / np.max(np.abs(signal))

# Save as WAV
write("input.wav", fs, signal.astype(np.float32))
print("Created input.wav")
