import argparse
from scipy.io import wavfile
import numpy as np

def parse_arguments():

    """Parse command line arguments for feedback suppression."""
    parser = argparse.ArgumentParser(
        description="Automatic feedback suppression using adaptive notch filters"
    )

    parser.add_argument("--in", dest="input_file", required = True, help="Input WAV file")
    parser.add_argument("--out", dest="output_file", required = True, help="Output WAV file")
    parser.add_argument("--threshold", dest="threshold", type=float, default=12.0, help="dB threshold for feeback detection (default : 12)")
    parser.add_argument("--max_notches", dest="max_notches", type=int, default=3, help="Maximum number of notches to apply (default : 3)")
    parser.add_argument("--q", type=float, default =30.0, help="Q factor for the notch filters (default : 30)")

    return parser.parse_args()

def load_audio(filename):
    """Load audio file and return sample rate and data.
    :param filename: Path to the input WAV file.
    :return: Sample rate and audio data as a numpy array.
    """
    fs, data = wavfile.read(filename) # Use the wavfile.read function from the scipy.io module to read the input WAV file specified by the filename parameter. This function returns the sample rate (fs) and the audio data (data) as a numpy array.
    if data.ndim > 1:
        data = data.mean(axis=1) # If the audio data has more than one channel (e.g., stereo), it averages the channels to convert it to mono.
    data = data.astype(np.float32) / 32768.0 # The audio data is then normalized to the range [-1, 1] by dividing it by 32768.0 (the maximum value for 16-bit audio).
    return fs, data # Finally, the function returns the sample rate and the normalized audio data as a tuple.

def save_audio(filename, fs, data):
    """Save audio data to a WAV file.
    :param filename: Path to the output WAV file.
    :param fs: Sample rate of the audio data.
    :param data: Audio data as a numpy array.
    """
    data_int16 = (data * 322768).asytype(np.int16) # The audio data is first scaled back to the range of 16-bit integers by multiplying it by 32768 and then converting it to the int16 data type.
    wavfile.write(filename, fs, data_int16) # The wavfile.write function from the scipy.io module is used to write the audio data to the specified output WAV file, using the provided sample rate (fs) and the converted audio data (data_int16).
    return
