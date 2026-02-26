#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audio_ext.py - Automatic Feedback Suppression using Adaptive Notch Filters
=======================================================================

This implements "Automatic feedback suppression: Detect and suppress feedback 
frequencies using adaptive notch filters" as specified in CM2208 Task 1 Extended 
Functionality.

METHOD OVERVIEW
---------------
Feedback (e.g., microphone → speaker → microphone loops) manifests as narrow, 
stable tonal peaks in the spectrum. This algorithm:

1. Processes audio in overlapping frames (1024 samples, 50% hop).
2. For each frame:
   - Applies Hann window + real FFT to estimate spectrum
   - Converts to dB, uses median as background floor
   - Finds local peaks > threshold_db above median (likely feedback)
   - Restricts to 200-8000 Hz (typical feedback range)
3. Maintains up to max_notches adaptive IIR notch filters:
   - If detected peak ≈ existing notch (±20 Hz), reset its age
   - Else if capacity available, design new notch at that frequency
   - Age notches each frame; prune if > max_age frames old
4. Applies cascade of active notches to current frame via manual biquad IIR

WHY THIS APPROACH?
------------------
IIR Biquad Notches vs Alternatives:
- IIR: Very efficient (3 mult-adds/sample), arbitrarily narrow notches (high Q)
  vs FIR: Linear phase but requires 100s-1000s taps for equivalent narrowness
  → IIR chosen for real-time capability and minimal collateral damage
- FFT-domain subtraction: Handles wideband but destroys phase/transients
  → Time-domain IIR preserves audio quality outside notch
- Peak detection via median+threshold: Robust to music (which has broad peaks)
  vs fixed frequencies: Adaptive to changing feedback paths

PERFORMANCE & LIMITATIONS
------------------------
✓ Strengths: Real-time capable, preserves audio quality, adapts to changing 
   feedback, handles multiple simultaneous tones
✗ Limitations: 
   - May mis-detect loud musical notes/harmonics as feedback
   - High-Q notches can cause slight pre-ringing (IIR phase nonlinearity)
   - Frame-based → ~25ms latency minimum
   - Works best for stable, narrow feedback; struggles with flutter/howling

USAGE EXAMPLES
--------------
$ python audio_ext.py --in noisy_feedback.wav --out clean.wav
$ python audio_ext.py --in input.wav --out output.wav --threshold 15 --max_notches 4 --q 40
$ python audio_ext.py --in mic_test.wav --out suppressed.wav --threshold 10 --q 50

ARGUMENTS
---------
--in, --out     : Required input/output WAV files
--threshold     : dB above background to detect feedback (default: 12.0)
--max_notches   : Max simultaneous notches (default: 3)
--q             : Notch Q factor (default: 30.0, higher=narrower)

All parameters have sensible defaults. Robust error handling included.

CM2208 Task 1 Extended Functionality - Submitted by Shaan Nagar
"""

import argparse
import numpy as np
from scipy.io import wavfile
from scipy.fft import rfft


def parse_arguments():
    """Parse command line arguments for feedback suppression."""
    parser = argparse.ArgumentParser(
        description="Automatic feedback suppression using adaptive notch filters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python audio_ext.py --in noisy.wav --out clean.wav
  python audio_ext.py --in input.wav --out output.wav --threshold 15 --q 40
        """
    )
    parser.add_argument("--in", dest="input_file", required=True, 
                       help="Input WAV file (required)")
    parser.add_argument("--out", dest="output_file", required=True, 
                       help="Output WAV file (required)")
    parser.add_argument("--threshold", dest="threshold", type=float, default=12.0,
                       help="dB threshold for feedback detection (default: 12)")
    parser.add_argument("--max_notches", dest="max_notches", type=int, default=3,
                       help="Maximum number of notches to apply (default: 3)")
    parser.add_argument("--q", type=float, default=30.0,
                       help="Q factor for notch filters (default: 30, higher=narrower)")
    return parser.parse_args()


def load_audio(filename):
    """Load WAV file and return sample rate + normalized mono signal."""
    try:
        fs, data = wavfile.read(filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file '{filename}' not found")
    except Exception as e:
        raise ValueError(f"Cannot read '{filename}': {e}")

    # Handle stereo → mono averaging
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    
    # Normalize 16-bit int → float [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
        print("Warning: unexpected audio format, treating as float")

    if len(data) == 0:
        raise ValueError("Audio file is empty")
        
    return fs, data


def save_audio(filename, fs, data):
    """Save normalized float signal → 16-bit WAV."""
    # Clip and scale to int16 range
    data = np.clip(data, -1.0, 1.0)
    data_int16 = (data * 32767).astype(np.int16)
    wavfile.write(filename, fs, data_int16)


def design_notch_filter(f0, fs, Q=30.0):
    """
    Design 2nd-order IIR notch filter (RBJ Audio EQ cookbook).

    IIR chosen over FIR because:
    - FIR needs 100s-1000s taps for narrow notch → high CPU
    - IIR biquad: 5 mult-adds/sample, infinite Q possible
    - Acceptable phase distortion for feedback suppression

    Parameters:
    -----------
    f0, fs, Q : see main docstring
    """
    w0 = 2.0 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2.0 * Q)

    b0, b1, b2 = 1.0, -2.0*np.cos(w0), 1.0
    a0, a1, a2 = 1.0+alpha, -2.0*np.cos(w0), 1.0-alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1/a0, a2/a0])
    return b, a


def detect_feedback_frequencies(frame, fs, threshold_db=12.0, max_notches=3):
    """
    FFT-based feedback detection via spectral peak picking.
    
    Median background + local peak test is robust to music (broad spectrum)
    vs fixed thresholds which fail on quiet/loud passages.
    """
    N = len(frame)
    if N < 128:
        return []

    # Hann window → FFT → magnitude spectrum
    window = np.hanning(N)
    X = rfft(frame * window)
    mag = np.abs(X) + 1e-12
    mag_db = 20 * np.log10(mag)

    # Median = robust background floor estimate
    median_db = np.median(mag_db)

    candidates = []
    for k in range(1, len(mag_db)-1):
        # Local peak AND threshold above background
        if (mag_db[k] > median_db + threshold_db and
            mag_db[k] > mag_db[k-1] and 
            mag_db[k] > mag_db[k+1]):
            
            f = k * fs / (2.0 * (len(mag_db) - 1))
            if 200.0 <= f <= 8000.0:  # Audible feedback range
                candidates.append((mag_db[k], f))

    # Return strongest peaks only
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [f for _, f in candidates[:max_notches]]


def iir_filter(x, b, a):
    """Manual 2nd-order IIR biquad (Direct Form II)."""
    x = np.asarray(x, dtype=np.float64)
    b, a = np.asarray(b), np.asarray(a)
    y = np.zeros_like(x)

    x1, x2 = 0.0, 0.0  # x[n-1], x[n-2]
    y1, y2 = 0.0, 0.0  # y[n-1], y[n-2]

    for n in range(len(x)):
        x0 = x[n]
        y0 = (b[0]*x0 + b[1]*x1 + b[2]*x2 - 
              a[1]*y1 - a[2]*y2)
        
        x2, x1 = x1, x0
        y2, y1 = y1, y0
        y[n] = y0

    return y.astype(x.dtype)


def apply_cascade_filters(x, coeffs_list):
    """Cascade multiple biquads in series."""
    y = x.copy()
    for b, a in coeffs_list:
        y = iir_filter(y, b, a)
    return y


def feedback_suppression(x, fs, threshold_db=12.0, max_notches=3, Q=30.0):
    """
    Frame-based adaptive feedback suppression (see module docstring).
    """
    frame_size, hop_size, max_age = 1024, 512, 30
    N = len(x)
    y = np.zeros_like(x)
    active_notches = []

    for start in range(0, N, hop_size):
        end = min(start + frame_size, N)
        frame = x[start:end]

        if len(frame) < 128:
            y[start:end] = frame
            continue

        # Detect feedback
        feedback_freqs = detect_feedback_frequencies(
            frame, fs, threshold_db, max_notches
        )

        # Update/create notches
        for f0 in feedback_freqs:
            found = False
            for notch in active_notches:
                if abs(notch['f0'] - f0) < 20.0:
                    notch['age'] = 0
                    found = True
                    break
            
            if not found and len(active_notches) < max_notches:
                b, a = design_notch_filter(f0, fs, Q)
                active_notches.append({'f0': f0, 'b': b, 'a': a, 'age': 0})

        # Age and prune
        for notch in active_notches:
            notch['age'] += 1
        active_notches = [n for n in active_notches if n['age'] < max_age]

        # Apply cascade
        coeffs_list = [(n['b'], n['a']) for n in active_notches]
        processed = (apply_cascade_filters(frame, coeffs_list) 
                    if coeffs_list else frame)
        y[start:end] = processed

    return np.clip(y, -1.0, 1.0)


if __name__ == "__main__":
    try:
        args = parse_arguments()
        
        print(f"Loading {args.input_file}...")
        fs, x = load_audio(args.input_file)
        duration = len(x) / fs
        print(f"Audio: {duration:.1f}s @ {fs}Hz, {len(x)} samples")
        
        # Parameter validation
        if args.threshold < 3.0 or args.threshold > 40.0:
            print("Warning: threshold outside typical range [3,40]")
        if args.q < 10.0 or args.q > 100.0:
            print("Warning: Q outside typical range [10,100]")
            
        print("Running adaptive feedback suppression...")
        y = feedback_suppression(
            x, fs,
            threshold_db=args.threshold,
            max_notches=args.max_notches,
            Q=args.q
        )
        
        print(f"Saving {args.output_file}...")
        save_audio(args.output_file, fs, y)
        print("Done! Feedback suppressed.")
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Usage: python audio_ext.py --in input.wav --out output.wav")
    except Exception as e:
        print(f"ERROR: {e}")
