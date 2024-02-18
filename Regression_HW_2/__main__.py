# Multivariate Linear Regression For Boston Housing Data
# AUTHORS: Garrett Thrower and Megan Jenckes
# Last Modified: 2/17/2024

# This script calculates the coefficients for a multivariate linear regression model using the least squares method.
# All done manually, since the point is to understand the math behind it.

from time import sleep
import matplotlib.pyplot as plt
import re
import os


def preprocess_data(file_path):
    """
    Reads a text file, removes every odd-line newline character, 
    and replaces multi-character whitespace with a single space.

    Parameters:
    - file_path: The path to the text file to be processed.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Process each line
    new_lines = []
    for i, line in enumerate(lines):
        if i % 2 == 0:  # Check if the line number is odd (considering 0-indexing)
            # Remove newline character and trailing spaces, then add to new_lines without a newline
            new_lines.append(line.strip())
        else:
            new_lines[-1] += re.sub(r'\s+', ' ', line).strip() + " "
    
    # Write the processed lines back to the file
    with open(file_path, 'w') as file:
        for line in new_lines:
            file.write(line + "\n")  # Add back a newline character at the end of each merged line
            
    # now that the file is cleaned up, rename it to its original name + "_cleaned.txt" so the program doesn't keep trying to reformat it.
    # first, we have to get the original filename without the extension
    filename = os.path.basename(file_path)
    filename = filename.split(".")[0]
    cleaned_filename = filename + "_cleaned.txt"
    cleaned_path = os.path.join(os.path.dirname(file_path), cleaned_filename)
    os.rename(file_path, cleaned_path)
    
    print("Data has been cleaned and saved to", cleaned_path)
    return cleaned_path
    
# Get the path to the user's desktop
user_desktop = os.path.join(os.path.expanduser("~"), "Desktop")

# Define the file paths
file_path = os.path.join(user_desktop, "boston.txt")
cleaned_file_path = os.path.join(user_desktop, "boston_cleaned.txt")

# Check if the cleaned file exists
if os.path.exists(cleaned_file_path):
    # Load the cleaned data
    cleaned_dataset = cleaned_file_path
else:
    # Check if the raw file exists
    if os.path.exists(file_path):
        # Preprocess the data and load it
        cleaned_dataset = preprocess_data(file_path)
    else:
        print("Neither data file was found.")
        

# at this point, the file path to the cleaned data should be stored in the variable cleaned_dataset.
# we can now load the data from the file and begin processing it.