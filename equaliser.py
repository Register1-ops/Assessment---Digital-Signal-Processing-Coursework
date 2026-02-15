import argparse
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def parse_arguments(): # This function is responsible for parsing the command-line arguments provided by the user when running the script. It uses the argparse library to define and handle these arguments.
    parser = argparse.ArgumentParser(
        description="11-band octave equaliser implemented using cascaded IIR peaking filters."
    ) # Define command-line arguments, when the user enters python equaliser.py --help, 
    #this description will be shown along with the list of arguments.

    parser.add_argument("--in", dest="input_file", required=True) # Define the --in argument, which is required and will be stored in the input_file variable.
    parser.add_argument("--out", dest="output_file", required=True) # Deines the --out argument, which is required from the user and will be stored in the output_file variable.
    parser.add_argument("--plot", dest="plot_file", required=True) # Defines the --plot argument, which is required from the user and will be stored in the plot_file variable.
    parser.add_argument("--bands", required=True) # Defines the --bands argument, which is required from the user. This argument will be used to specify the gain values for each of the 11 bands of the equaliser.

    return parser.parse_args()


def parse_bands(bands_str): # This function takes the string of comma-separated values provided by the user for the --bands argument and processes it to extract the gain values for each band.
    values = [float(x)  for x in bands_str.split(",")] # This function takes a string of comma-separated values (representing the gain values for each band) and converts it into a list of floating-point numbers.
    if len(values) != 11: # Check if the number of values provided is exactly 11, which is the expected number of bands for the equaliser. If not, it raises a ValueError.
        raise ValueError("Exactly 11 band values must be provided.")
    return values # Return the list of gain values for the 11 bands.

def db_to_amplitude(db): # This function converts a gain value from decibels (dB) to amplitude. The formula used is: amplitude = 10^(db/20).
    return 10**(db/20) # This function takes a gain value in decibels (dB) and converts it to amplitude using the formula: amplitude = 10^(db/20). This is a common conversion in audio processing, as it allows us to work with linear amplitude values instead of logarithmic dB values.

def design_peaking_filter(fc, fs, gain_db, Q = 1.0): # This function designs a peaking filter based on the specified center frequency (fc), sampling frequency (fs), gain in decibels (gain_db), and quality factor (Q). It calculates the filter coefficients for the peaking filter and returns them as two arrays: b for the numerator coefficients and a for the denominator coefficients.
    
    A = db_to_amplitude(gain_db) #' Convert the gain from decibels to amplitude using the db_to_amplitude function.
    w0 = 2 * np.pi * ( fc / fs ) # Calculate the normalized angular frequency (w0) for the peaking filter based on the center frequency (fc) and the sampling frequency (fs).
    alpha  = np.sin(w0) / (2 * Q) # Calculate the alpha parameter for the peaking filter, which is based on the normalized angular frequency (w0) and the quality factor (Q).

    b0 = 1 + alpha * A # Calculate the b0 coefficient for the peaking filter using the gain (A) and the alpha parameter.
    b1 = -2 * np.cos(w0) # Calculate the b1 coefficient for the peaking filter using the normalized angular frequency (w0).
    b2 = 1 - alpha * A # Calculate the b2 coefficient for the peaking filter using the gain (A) and the alpha parameter.
    
    a0 = 1 + alpha / A # Calculate the a0 coefficient for the peaking filter using the gain (A) and the alpha parameter.
    a1 = -2 * np.cos(w0) # Calculate the a1 coefficient for the peaking filter using the normalized angular frequency (w0).
    a2 = 1 - alpha / A # Calculate the a2 coefficient for the peaking filter using the gain (A) and the alpha parameter.

    # Normalize the coefficients by a0 to ensure that the filter has a gain of 1 at the center frequency when the gain is set to 0 dB.

    b = np.array([b0, b1, b2]) / a0 # Normalize the b coefficients by a0.
    a = np.array([1, a1 / a0, a2 / a0]) # Normalize the a coefficients by a0, and set the first coefficient to 1 for the standard IIR filter representation.

    return b, a # Return the normalized filter coefficients b and a.

def load_audio(filename):
    audioData, fs = sf.load(filename) # Load the audio file specified by the filename using the soundfile library. This function returns the audio data as a NumPy array and the sampling frequency (fs) of the audio.
    if audioData.ndim > 1: # Check if the audio data has more than one channel (i.e., is stereo). If it does, convert it to mono by averaging the channels.
        audioData = audioData.mean(axis=1) # Convert stereo audio to mono by averaging the channels.
    return audioData, fs # Return the loaded audio data and the sampling frequency.

def save_audio(filename, audioData, fs):
    """Save audio files"""
    sf.write(filename, audioData, fs) # Save the processed audio data to a file specified by the filename using the soundfile library. The audio data is saved with the given sampling frequency (fs).
    return

def build_equaliser(fs, gains_db): # This function builds the equaliser by designing peaking filters for each of the 11 bands based on the specified sampling frequency (fs) and the gain values in decibels (gains_db). It returns a list of filter coefficients for each band.
    """
    Docstring for build_equaliser
    
    :param fs: 
    :param gains_db: 
    """
    center_frequencies = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000] # Define the center frequencies for the 11 bands of the equaliser.
    coefficients = [] # Initialize an empty list to store the filter coefficients for each band.

    for fc, gain_db in zip(center_frequencies, gains_db): # Loop through each center frequency and corresponding gain value in decibels, and design a peaking filter for each band using the design_peaking_filter function. The resulting filter coefficients are appended to the coefficients list.
        if fc >= fs / 2: # Check if the center frequency (fc) is greater than or equal to half of the sampling frequency (fs). If it is, skip designing the filter for that band, as it would not be valid due to the Nyquist frequency limit.
            continue
        b, a = design_peaking_filter(fc, fs, gain_db) # Design a peaking filter for the current center frequency (fc) and gain in decibels (gain_db) using the design_peaking_filter function. This function returns the filter coefficients b and a.
        coefficients.append((b, a)) # Append the filter coefficients (b, a) as a tuple to the coefficients list.
    return coefficients # Return the list of filter coefficients for each band of the equaliser.

if __name__ == "__main__":
    args = parse_arguments()
