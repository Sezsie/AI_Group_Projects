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

""" 
For the sake of convenience, the names of the classes of the iris dataset are as follows:
    1. Iris-setosa
    2. Iris-versicolor
    3. Iris-virginica
"""


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

# cost function for logistic regression
def cost_function(thetas, x, y):
    """
    The cost function for logistic regression
    """
    # number of training examples
    m = len(y)
    # calculate the hypothesis
    hypothesis = sigmoid(np.dot(x, thetas))
    # calculate the cost
    cost = (1 / m) * np.sum(-y * np.log(hypothesis) - (1 - y) * np.log(1 - hypothesis))
    return cost

# gradient descent for the logistic regression
def gradient_descent(thetas, x, y, alpha, iterations):
    """
    Gradient descent for logistic regression
    """
    # number of training examples
    m = len(y)
    # loop through the number of iterations
    for _ in range(iterations):
        # calculate the hypothesis
        hypothesis = sigmoid(np.dot(x, thetas))
        # calculate the gradient
        gradient = (1 / m) * np.dot(x.T, (hypothesis - y))
        # update the thetas
        thetas -= alpha * gradient
    return thetas

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

#print("The training set:")
#print(training_set)
#print("The validation set:")
#print(validation_set)

# Extract x features into an array
training_set_x = np.array([datapoint[:-1] for datapoint in training_set])

# Extract y feature into an array
training_set_y = np.array([int(datapoint[-1] == 'Iris-setosa') for datapoint in training_set])

# Extract x features into an array
validation_set_x = np.array([datapoint[:-1] for datapoint in validation_set])

# Extract y feature into an array and convert to integers
validation_set_y = np.array([int(datapoint[-1] == 'Iris-setosa') for datapoint in validation_set])


# Print the first few elements of each list to verify
print("Training set x:", training_set_x[:5], "\n")
print("Classifications:", [datapoint[-1] for datapoint in training_set[:5]], "\n")
print("Training set y:", training_set_y[:5], "\n")

# THIS IS FOR LATER: Adding an intercept
training_set_x = np.insert(training_set_x, 0, 1, axis=1)  # Add intercept term
validation_set_x = np.column_stack([np.ones(validation_set_x.shape[0]), validation_set_x])

# Initial theta values
initial_thetas = np.zeros(training_set_x.shape[1])  # Shape[1] gives the number of columns (features)

# Testing the sigmoid function
result = sigmoid(0)
print("Sigmoid:", result)

# part 2, we will implement the cost function and gradient descent for logistic regression

def cost_function(thetas, x, y):
    """
    The cost function for logistic regression
    """
    # number of training examples
    m = len(y)
    # calculate the hypothesis
    hypothesis = sigmoid(np.dot(x, thetas))
    # calculate the cost
    cost = (1 / m) * np.sum(-y * np.log(hypothesis) - (1 - y) * np.log(1 - hypothesis))
    return cost


# part 2, we will implement the cost function and gradient descent for logistic regression
# test that we have implemented the cost function correctly, ideally the result should be 0.693 for the current dataset
cost = cost_function(initial_thetas, training_set_x, training_set_y)
print("Cost:", cost)

# test the gradient descent function
thetas = gradient_descent(initial_thetas, training_set_x, training_set_y, 0.01, 1000)
print("Thetas:", thetas)

