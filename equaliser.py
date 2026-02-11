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




if __name__ == "__main__":
    args = parse_arguments()
