import argparse
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import freqz

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
    audioData, fs = sf.read(filename) # Load the audio file specified by the filename using the soundfile library. This function returns the audio data as a NumPy array and the sampling frequency (fs) of the audio.
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

def apply_cascade_filters(x, coeffs_list):
    """Apply a cascade of IIR filters to signal x."""
    y = x.copy()
    for b, a in coeffs_list:
        y = np.convolve(y, b, mode='same')  # or use scipy.signal.lfilter if allowed
        # If you are allowed to use scipy.signal.lfilter:
        # from scipy.signal import lfilter
        # y = lfilter(b, a, y)
    return y



def iir_filter(x , b, a): # This function applies an IIR filter to the input signal x using the filter coefficients b (numerator) and a (denominator). It processes the input signal sample by sample, applying the filter equation to produce the output signal y.
    """Apply an IIR filter to the input signal x using the filter coefficients b and a."""
    y = np.zeros_like(x) # Initialize the output signal y as an array of zeros with the same shape as the input signal x.
    for n in range(len(x)): # Loop through each sample in the input signal x, applying the IIR filter equation to compute the corresponding output sample in y. The filter equation is based on the current and previous input samples (x) and the previous output samples (y), weighted by the filter coefficients b and a.
        y[n] = b[0]*x[n]# Apply the current input sample x[n] weighted by the first coefficient b[0] to the output y[n].
        if n >=1:   # If there is at least one previous sample, apply the first-order terms of the filter equation, which involve the previous input sample x[n-1] weighted by b[1] and the previous output sample y[n-1] weighted by a[1].
            y[n] += b[1]*x[n-1] - a[1]*y[n-1] # Apply the first-order terms of the filter equation, which involve the previous input sample x[n-1] weighted by b[1] and the previous output sample y[n-1] weighted by a[1].
        if n >=2: # If there are at least two previous samples, apply the second-order terms of the filter equation, which involve the input sample from two steps back x[n-2] weighted by b[2] and the output sample from two steps back y[n-2] weighted by a[2].
            y[n] += b[2]*x[n-2] - a[2]*y[n-2] # Apply the second-order terms of the filter equation, which involve the input sample from two steps back x[n-2] weighted by b[2] and the output sample from two steps back y[n-2] weighted by a[2].
    return y

def apply_cascade_filters(x, coeffs_list): # This function applies a cascade of IIR filters to the input signal x using a list of filter coefficients (coeffs_list). It iteratively applies each filter in the cascade to the output of the previous filter, starting with the original input signal x.
    y = x.copy() #  Initialize the output signal y as a copy of the input signal x. This will be updated iteratively as each filter in the cascade is applied.
    for b, a in coeffs_list:    # Loop through each set of filter coefficients (b, a) in the coeffs_list, applying the iir_filter function to the current output signal y using the current filter coefficients. The output of each filter becomes the input for the next filter in the cascade.
        y = iir_filter(y, b, a) #    Apply the current filter to the signal y using the iir_filter function, which processes the signal with the given filter coefficients b and a. The output of this filter becomes the new y for the next iteration of the loop.
    return y # Return the final output signal after applying all filters in the cascade.

def plot_freq_response(coeffs_list, fs, plot_file): # This function plots and saves the frequency response of the equaliser based on the list of filter coefficients (coeffs_list) and the sampling frequency (fs). It uses the freqz function to compute the frequency response of each filter in the cascade, multiplies them together to get the overall frequency response, and then plots it on a logarithmic scale. The resulting plot is saved to the specified plot_file.
    """Plot and save frequency response of the equaliser.""" # This is a docstring that describes the purpose of the plot_freq_response function, which is to plot and save the frequency response of the equaliser based on the provided filter coefficients and sampling frequency.
    w = np.linspace(0, np.pi, 2048) # Generate a range of frequencies (w) from 0 to π (Nyquist frequency) with 2048 points. This will be used to compute the frequency response of the filters.
    H = np.ones_like(w, dtype=complex) # Initialize the overall frequency response H as an array of ones (complex) with the same shape as w. This will be multiplied by the frequency response of each filter in the cascade to get the overall frequency response.
    for b, a in coeffs_list: # Loop through each set of filter coefficients (b, a) in the coeffs_list, compute the frequency response of the current filter using the freqz function, and multiply it with the overall frequency response H to accumulate the effect of all filters in the cascade.
        w_, h = freqz(b, a, worN=w) #   Compute the frequency response of the current filter using the freqz function, which takes the filter coefficients b and a, and the range of frequencies w. It returns the frequencies w_ (which should match w) and the corresponding frequency response h.
        H *= h # Multiply the frequency response of the current filter (h) with the overall frequency response H to accumulate the effect of all filters in the cascade. This results in the combined frequency response of the equaliser.

    f = w * fs / (2 * np.pi) # Convert the normalized frequencies w to actual frequencies in Hz by multiplying by the sampling frequency (fs) and dividing by 2π. This gives us the frequency values corresponding to the frequency response H.
    plt.figure(figsize=(10, 5)) # Create a new figure for plotting with a specified size of 10 inches by 5 inches.
    plt.semilogx(f, 20 * np.log10(np.abs(H))) # Plot the frequency response on a logarithmic scale for the x-axis (frequency) and a linear scale for the y-axis (gain in dB). The gain is calculated as 20 times the logarithm base 10 of the magnitude of the frequency response H.
    plt.xlim(20, 20000) # Set the limits for the x-axis to range from 20 Hz to 20,000 Hz, which is the typical range of human hearing.
    plt.xlabel("Frequency (Hz)") # Set the label for the x-axis to "Frequency (Hz)" to indicate that the x-axis represents frequency in Hertz.
    plt.ylabel("Gain (dB)") # Set the label for the y-axis to "Gain (dB)" to indicate that the y-axis represents gain in decibels.  
    plt.title("Equaliser Frequency Response")   # Set the title of the plot to "Equaliser Frequency Response" to describe what the plot represents.
    plt.grid(True, which="both", axis="both") # Add a grid to the plot for better visibility, with lines for both major and minor ticks on both axes.
    plt.savefig(plot_file) # Save the generated plot to the specified plot_file using the savefig function from matplotlib.
    plt.close() # Close the plot to free up memory and resources after saving it.
    return

if __name__ == "__main__":
    args = parse_arguments() # Parse the command-line arguments provided by the user when running the script. This will give us access to the input file, output file, plot file, and band gain values specified by the user.
 
    parsedBands = parse_bands(args.bands) # Parse the band gain values provided by the user for the --bands argument. This will convert the comma-separated string of gain values into a list of floating-point numbers that can be used to design the filters for the equaliser.

    loadedAudio = load_audio(args.input_file) # Load the audio file specified by the user using the load_audio function. This will return the audio data as a NumPy array and the sampling frequency (fs) of the audio, which will be used for designing the filters and processing the audio signal.
 
    coeffs_list = build_equaliser(loadedAudio[1], parsedBands) #    Build the equaliser by designing peaking filters for each of the 11 bands based on the sampling frequency (fs) of the loaded audio and the parsed band gain values. This will return a list of filter coefficients for each band, which will be used to process the audio signal.
 
    processedAudio = apply_cascade_filters(loadedAudio[0], coeffs_list) # Apply the cascade of IIR filters to the loaded audio signal using the apply_cascade_filters function. This will process the audio signal with the designed equaliser filters and produce the processed audio output.

    save_audio(args.output_file, processedAudio, loadedAudio[1]) #  Save the processed audio to a file specified by the user using the save_audio function. This will write the processed audio data to the output file with the same sampling frequency as the original audio.
    plot_freq_response(coeffs_list, loadedAudio[1], args.plot_file) # Plot and save the frequency response of the equaliser using the plot_freq_response function. This will generate a plot of the frequency response based on the filter coefficients and sampling frequency, and save it to the specified plot file.


