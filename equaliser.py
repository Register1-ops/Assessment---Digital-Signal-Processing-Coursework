#!/usr/bin/env python3
"""
equaliser.py - 11-Band Octave Graphic Equaliser (CM2208 Task 1)
================================================================

Implements 11 fixed octave bands (16-16kHz) using cascaded peaking 
IIR biquad filters per RBJ Audio EQ Cookbook. MANUAL IIR implementation.

METHOD
------
1. Parse 11 dB gains for exact center frequencies: [16,31.5,63,125,250,500,
   1000,2000,4000,8000,16000] Hz
2. Design peaking biquad for each band (manual coefficient calculation)
3. Apply cascade via MANUAL Direct-Form II IIR (no np.convolve/lfilter)
4. Plot combined frequency response (manual computation, no freqz)

WHY THIS APPROACH?
------------------
Peaking IIR Biquads vs alternatives:
- IIR: 5 mult-adds/sample per band vs FIR 100s taps → real-time capable
- Peaking: Boost/cut centered at fc vs shelving (only edges) → true graphic EQ
- Q=1.0: ~1 octave bandwidth matching band spacing → smooth transitions
vs FFT-domain: destroys phase/transients, not causal/real-time

PERFORMANCE
-----------
✓ Precise band gain control ✓ Smooth inter-band transitions 
✓ Real-time capable (55 mult-adds/sample total) ✓ Minimal phase distortion
✗ Slight IIR pre-ringing on steep boosts/cuts ✗ High fs needed for 16kHz band

USAGE
-----
python equaliser.py --in input.wav --out output.wav --plot resp.png --bands "0,0,3,0,-2,0,2,-1,0,1,0"
"""

import argparse
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="11-band octave graphic equaliser (manual IIR peaking filters)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python equaliser.py --in song.wav --out eqd.wav --plot resp.png --bands '0,3,0,-2,4,0,-1,2,0,-3,1'"
    )
    parser.add_argument("--in", dest="input_file", required=True, help="Input audio")
    parser.add_argument("--out", dest="output_file", required=True, help="Output audio") 
    parser.add_argument("--plot", dest="plot_file", required=True, help="Frequency response PNG")
    parser.add_argument("--bands", required=True, help="11 comma-separated dB gains")
    return parser.parse_args()

def parse_bands(bands_str):
    values = [float(x.strip()) for x in bands_str.split(",")]
    if len(values) != 11:
        raise ValueError(f"Exactly 11 band gains required, got {len(values)}")
    return values

def db_to_linear(db):
    """dB → amplitude multiplier."""
    return 10.0 ** (db / 20.0)

def design_peaking_filter(fc, fs, gain_db, Q=1.0):
    """
    RBJ Audio EQ Cookbook peaking filter coefficients.
    
    Peaking chosen for true band boost/cut centered at fc.
    Q=1.0 gives ~1 octave bandwidth matching band spacing.
    """
    A = db_to_linear(gain_db)
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * Q)
    
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0) 
    b2 = 1 - alpha * A
    
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A
    
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1/a0, a2/a0])
    return b, a

def load_audio(filename):
    """Load → mono float32 [-1,1]."""
    try:
        data, fs = sf.read(filename)
        if data.ndim > 1:
            data = np.mean(data, axis=1, dtype=np.float32)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        return data, fs
    except Exception as e:
        raise ValueError(f"Cannot load {filename}: {e}")

def save_audio(filename, data, fs):
    """Float32 [-1,1] → 16-bit WAV."""
    data = np.clip(data, -1.0, 1.0)
    data_int16 = (data * 32767).astype(np.int16)
    sf.write(filename, data_int16, fs)

def manual_iir_filter(x, b, a):
    """
    MANUAL 2nd-order Direct Form II IIR - NO LIBRARY FILTERING.
    
    y[n] = b[0]*x[n] + b[1]*x[n-1] + b[2]*x[n-2]
         - a[1]*y[n-1] - a[2]*y[n-2]
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x)
    
    # Delay lines
    x1, x2 = 0.0, 0.0
    y1, y2 = 0.0, 0.0
    
    for n in range(len(x)):
        x0 = x[n]
        y0 = (b[0]*x0 + b[1]*x1 + b[2]*x2 - 
              a[1]*y1 - a[2]*y2)
        
        x2, x1 = x1, x0
        y2, y1 = y1, y0
        y[n] = y0
    
    return y.astype(np.float32)

def apply_equaliser(x, coeffs_list):
    """Cascade all 11 band filters."""
    y = x.copy()
    for b, a in coeffs_list:
        y = manual_iir_filter(y, b, a)
    return y

def manual_freq_response(coeffs_list, fs, num_points=8192):
    """
    MANUAL frequency response computation - NO scipy.signal.freqz.
    
    Evaluate H(w) = B(w)/A(w) across log frequency points.
    """
    # Log spaced frequencies 20Hz → Nyquist
    f = np.logspace(np.log10(20), np.log10(fs/2), num_points)
    w = 2 * np.pi * f / fs  # normalized angular freq
    
    H = np.ones(num_points, dtype=complex)
    
    # Cascade all filters: H_total = H1 * H2 * ... * H11
    for b, a in coeffs_list:
        H_num = b[0] + b[1]*np.exp(-1j*w) + b[2]*np.exp(-2j*w)
        H_den = 1.0   + a[1]*np.exp(-1j*w) + a[2]*np.exp(-2j*w)
        H *= H_num / H_den
    
    return f, 20*np.log10(np.abs(H) + 1e-12)

def plot_frequency_response(coeffs_list, fs, plot_file):
    """Plot/save combined EQ response."""
    f, mag_db = manual_freq_response(coeffs_list, fs)
    
    plt.figure(figsize=(12, 6))
    plt.semilogx(f, mag_db, linewidth=2)
    plt.xlim(20, 20000)
    plt.ylim(-30, 30)
    plt.grid(True, which="both", alpha=0.3)
    plt.axhline(0, color='k', alpha=0.3)
    plt.axvline(1000, color='gray', linestyle=':', alpha=0.5)
    
    # Mark band centers
    band_freqs = [16,31.5,63,125,250,500,1000,2000,4000,8000,16000]
    for fc in band_freqs:
        if fc <= fs/2:
            plt.axvline(fc, color='red', alpha=0.7, linestyle='--', linewidth=1)
    
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Gain (dB)")
    plt.title("11-Band Octave Equaliser - Combined Frequency Response")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

def build_equaliser(fs, gains_db):
    """Design 11 octave band peaking filters."""
    center_freqs = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    coeffs = []
    
    for fc, gain_db in zip(center_freqs, gains_db):
        if fc >= fs/2:  # Skip Nyquist+ bands
            continue
        b, a = design_peaking_filter(fc, fs, gain_db, Q=1.0)
        coeffs.append((b, a))
    
    return coeffs

if __name__ == "__main__":
    try:
        args = parse_arguments()
        bands = parse_bands(args.bands)
        
        print(f"Loading {args.input_file}...")
        x, fs = load_audio(args.input_file)
        print(f"Audio: {len(x)/fs:.2f}s @ {fs}Hz")
        
        print("Designing 11-band equaliser...")
        coeffs = build_equaliser(fs, bands)
        print(f"Applied gains: {bands}")
        
        print("Processing...")
        y = apply_equaliser(x, coeffs)
        
        print(f"Saving audio → {args.output_file}")
        save_audio(args.output_file, y, fs)
        
        print(f"Saving frequency plot → {args.plot_file}")
        plot_frequency_response(coeffs, fs, args.plot_file)
        
        print("Done!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("Usage: python equaliser.py --in in.wav --out out.wav --plot plot.png --bands '0,0,3,0,-2,...'")
