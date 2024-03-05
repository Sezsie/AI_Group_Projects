#-------------------------------------------------------------------------
# HW03: Logistic Regression For Iris Dataset
# CS430-01 Spring 2024
# Authors: Garrett Thrower and Megan Jenckes
# Date: March 8th, 2024
# IDE: Visual Studio Code version 1.86.0 January 2024
#
# File: _main_.py
#
# Purpose: 
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# Libraries needed to complete the program:
#-------------------------------------------------------------------------
import os
import numpy as np
import scipy
import random
from sigmoid import sigmoid

#-------------------------------------------------------------------------
# Functions:
#--------------------------------------------------------------------------

# the file path to the data should be stored in the variable file_path found below.
# we can now load the data from the file and begin processing it.
def persistent_load(file_path):
    """
    Loads a dataset line by line from a text file into a list of lists, where each entry is a string that relates to the data type.
    Once the algorithm hits a newline, it will assume that the next line is the start of a new data sample.
    """
    # open the file and read the lines
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Initialize empty lists to store data values
    data = []
    # Process each line
    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue
    
        values = line.strip().split(',')
        data_values = [float(val) for val in values[:4]]  # Extract the four features as floats
        data_values.append(values[4]) # add the y value as a string
        data.append(data_values)  # add the five values to the data list
    
    return data

# relative path to the data file, assuming it's on the desktop
user_desktop = os.path.join(os.path.expanduser("~"), "Desktop")
# set the file path for the original file
file_path = os.path.join(user_desktop, "iris.data")

#-------------------------------------------
# LOADING AND SPLITTING THE DATA PORTION
#-------------------------------------------

data = persistent_load(file_path)

# Print the data list to ensure it was processed correctly
# print(data)

# Shuffle the dataset
random.shuffle(data)

# Calculate at which index do we hit 80% of the dataset
index_80 = int(len(data) * 0.8)

# Split the dataset into two parts, the training and validation set
training_set = data[:index_80]
validation_set = data[index_80:]

print("The training set:")
print(training_set)
print("The validation set:")
print(validation_set)

# Extract x features into an array
training_set_x = np.array([datapoint[:-1] for datapoint in training_set])

# Extract y feature into an array
training_set_y = np.array([datapoint[-1] for datapoint in training_set])

# Extract x features into an array
validation_set_x = np.array([datapoint[:-1] for datapoint in validation_set])

# Extract y feature into an array
validation_set_y = np.array([datapoint[-1] for datapoint in validation_set])

# Print the first few elements of each list to verify
print("Training set x:", training_set_x[:5])
print("Training set y:", training_set_y[:5])

# Initial theta values
initial_thetas = np.zeros(len(training_set_x) + 1) # +1 for the intercept

# THIS IS FOR LATER: Adding an intercept
training_set_x = np.column_stack([np.ones(training_set_x.shape[0]), training_set_x])
validation_set_x = np.column_stack([np.ones(validation_set_x.shape[0]), validation_set_x])

# Testing the sigmoid function
result = sigmoid(0)
print("Sigmoid:", result)