import argparse
import numpy as np
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq


def parse_arguments():
    """Parse command line arguments for feedback suppression."""
    parser = argparse.ArgumentParser(
        description="Automatic feedback suppression using adaptive notch filters"
    )
    parser.add_argument("--in", dest="input_file", required=True, help="Input WAV file")
    parser.add_argument("--out", dest="output_file", required=True, help="Output WAV file")
    parser.add_argument("--threshold", dest="threshold", type=float, default=12.0,
                        help="dB threshold for feedback detection (default: 12)")
    parser.add_argument("--max_notches", dest="max_notches", type=int, default=3,
                        help="Maximum number of notches to apply (default: 3)")
    parser.add_argument("--q", type=float, default=30.0,
                        help="Q factor for the notch filters (default: 30)")
    return parser.parse_args()


def load_audio(filename):
    """Load audio file and return sample rate and data."""
    fs, data = wavfile.read(filename)          # Read WAV: fs = sample rate, data = samples
    if data.ndim > 1:
        data = data.mean(axis=1)               # Convert stereo to mono by averaging channels
    data = data.astype(np.float32) / 32768.0   # Normalize 16‑bit int to float in [-1, 1]
    return fs, data                            # Return sample rate and normalized signal


def save_audio(filename, fs, data):
    """Save audio data to a WAV file."""
    data_int16 = (data * 32768).astype(np.int16)  # Scale back to 16‑bit integer range
    wavfile.write(filename, fs, data_int16)       # Write WAV file with given sample rate


def design_notch_filter(f0, fs, Q=30.0):
    """Design a narrow notch filter at frequency f0."""
    w0 = 2 * np.pi * f0 / fs                    # Normalized angular frequency (radians/sample)
    alpha = np.sin(w0) / (2 * Q)                # Bandwidth parameter (controls notch width)

    b0 = 1                                      # Numerator coefficient 0
    b1 = -2 * np.cos(w0)                        # Numerator coefficient 1
    b2 = 1                                      # Numerator coefficient 2

    a0 = 1 + alpha                              # Denominator coefficient 0 (normalization)
    a1 = -2 * np.cos(w0)                        # Denominator coefficient 1
    a2 = 1 - alpha                              # Denominator coefficient 2

    b = np.array([b0, b1, b2]) / a0            # Normalize numerator
    a = np.array([1, a1 / a0, a2 / a0])        # Normalize denominator (a[0] = 1)
    return b, a                                 # Return biquad coefficients


def detect_feedback_frequencies(frame, fs, threshold_db=12.0, max_notches=3):
    """Detect potential feedback frequencies in a frame."""
    N = len(frame)
    if N < 128:
        return []                               # Skip very short frames

    window = np.hanning(N)                      # Apply Hann window to reduce spectral leakage
    X = rfft(frame * window)                    # Real FFT: spectrum from 0 to Nyquist
    mag = np.abs(X) + 1e-12                     # Magnitude (add small offset to avoid log(0))
    mag_db = 20 * np.log10(mag)                 # Convert to decibels

    median_db = np.median(mag_db)               # Estimate background level

    candidates = []
    for k in range(1, len(mag_db) - 1):         # Loop over FFT bins
        if (mag_db[k] > median_db + threshold_db and
            mag_db[k] > mag_db[k-1] and
            mag_db[k] > mag_db[k+1]):           # Local peak above threshold

            f = k * fs / (2 * (len(mag_db) - 1))  # Convert bin index to frequency (Hz)
            if 200 <= f <= 8000:                # Only keep audible feedback range
                candidates.append((mag_db[k], f))

    candidates.sort(reverse=True, key=lambda x: x[0])  # Sort by magnitude (loudest first)
    return [f for _, f in candidates[:max_notches]]    # Return top frequencies


def iir_filter(x, b, a):
    """Apply a 2nd‑order IIR filter (biquad) to signal x."""
    y = np.zeros_like(x)                        # Output buffer
    x_delay1, x_delay2 = 0, 0                   # x[n-1], x[n-2]
    y_delay1, y_delay2 = 0, 0                   # y[n-1], y[n-2]

    for n in range(len(x)):
        x_curr = x[n]
        y_curr = (b[0] * x_curr +
                  b[1] * x_delay1 +
                  b[2] * x_delay2 -
                  a[1] * y_delay1 -
                  a[2] * y_delay2)              # Direct‑form II IIR equation

        x_delay2 = x_delay1
        x_delay1 = x_curr
        y_delay2 = y_delay1
        y_delay1 = y_curr
        y[n] = y_curr

    return y


def apply_cascade_filters(x, coeffs_list):
    """Apply cascade of IIR filters: x -> filter1 -> filter2 -> ..."""
    y = x.copy()                                # Start with original signal
    for b, a in coeffs_list:
        y = iir_filter(y, b, a)                 # Chain filters in series
    return y


def feedback_suppression(x, fs, threshold_db=12.0, max_notches=3, Q=30.0):
    """Main feedback suppression algorithm (frame‑based)."""
    frame_size = 1024                           # FFT frame size
    hop_size = 512                              # Hop between frames
    max_age = 30                                # Frames before notch expires

    y = np.zeros_like(x)                        # Output signal buffer
    active_notches = []                         # List of active notch filters

    for start in range(0, len(x), hop_size):
        end = min(start + frame_size, len(x))
        frame = x[start:end]

        if len(frame) < 128:
            y[start:end] = frame
            continue

        # 1. Detect feedback frequencies in this frame
        feedback_freqs = detect_feedback_frequencies(
            frame, fs, threshold_db, max_notches
        )

        # 2. Update or create notch filters
        for f0 in feedback_freqs:
            found = False
            for notch in active_notches:
                if abs(notch['f0'] - f0) < 20:   # Within 20 Hz (similar frequency)
                    notch['age'] = 0             # Reset age (keep it alive)
                    found = True
                    break

            if not found and len(active_notches) < max_notches:
                b, a = design_notch_filter(f0, fs, Q)
                active_notches.append({'f0': f0, 'b': b, 'a': a, 'age': 0})

        # 3. Age and prune old notches
        for notch in active_notches:
            notch['age'] += 1
        active_notches = [n for n in active_notches if n['age'] < max_age]

        # 4. Apply active notches to this frame
        coeffs_list = [(n['b'], n['a']) for n in active_notches]
        if coeffs_list:
            processed = apply_cascade_filters(frame, coeffs_list)
        else:
            processed = frame

        y[start:end] = processed

    return y


if __name__ == "__main__":
    args = parse_arguments()

    print(f"Loading {args.input_file}...")
    fs, x = load_audio(args.input_file)         # Load audio: fs, x (not x, fs)
    print(type(x), x.shape)                     # Should be <class 'numpy.ndarray'> and (N,)
    print(f"Audio: {len(x)/fs:.1f}s @ {fs}Hz")  # Print duration and sample rate

    print("Running feedback suppression...")
    y = feedback_suppression(
        x, fs,
        threshold_db=args.threshold,
        max_notches=args.max_notches,
        Q=args.q
    )

    print(f"Saving {args.output_file}...")
    save_audio(args.output_file, fs, y)         # Save clean audio (filename, fs, data)
    print("Done!")
