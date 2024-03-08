#-------------------------------------------------------------------------
# HW03: Logistic Regression For Iris Dataset
# CS430-01 Spring 2024
# Authors: Garrett Thrower and Megan Jenckes
# Date: March 8th, 2024
# IDE: Visual Studio Code version 1.86.0 January 2024
#
# File: _main_.py
#
# Purpose: Implement and test logistic regression by using the cost and gradient
# function to find the parameters for logistic regression using the dataset from the
# iris.data file. Implement and find the confusion matrix by making predictions
# using the optimal parameters found by the Broyden-Fletcher-Goldfarb-Shanno 
# optimization algorithm from the scipy library and calculate the accuracy and precision. 
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# Libraries needed to complete the program:
#-------------------------------------------------------------------------
import os
import numpy as np
from scipy.optimize import minimize
from sigmoid import sigmoid
from sklearn.model_selection import train_test_split

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
    # clip the hypothesis to avoid 0 and 1 (it gets unhappy when log is 0 or 1 so...)
    hypothesis = np.clip(hypothesis, 1e-15, 1 - 1e-15)
    # calculate the cost
    cost = (1 / m) * np.sum(-y * np.log(hypothesis) - (1 - y) * np.log(1 - hypothesis))
    return cost

# gradient function for the logistic regression
def gradient(thetas, x, y):
    """
    Gradient for logistic regression
    """
    # number of training examples
    m = len(y)
    # calculate the hypothesis
    hypothesis = sigmoid(np.dot(x, thetas))
    # calculate the gradient
    gradient = (1 / m) * np.dot(x.T, (hypothesis - y))
    
    return gradient

# prediction function for the logistic regression
def predict(x, theta):
    """
    Predict the class label for each sample in x using the optimal thetas.
    
    Parameters:
    - x: Input features 
    - theta: the optimal thetas
    
    Returns:
    - predictions: Predicted class labels 
    """
    # Calculate the predicted probability
    probability = 1 / (1 + np.exp(-np.dot(x, theta)))  # P(y = 1| x; theta)
    
    # Convert probability to binary predictions
    predictions = (probability >= 0.5).astype(int)
    
    return predictions

# confusion matrix function for the logistic regression
def confusion_matrix(predictions, y):
    """
    Creates the confusion matrix 
    
    Parameters:
    - predictions: predictions made based on the optimal thetas and x values
    - y: the 
    
    Returns:
    - predictions: Predicted class labels 
    """
    TP = np.sum((predictions == 1) & (y == 1))
    FP = np.sum((predictions == 1) & (y == 0))
    TN = np.sum((predictions == 0) & (y == 0))
    FN = np.sum((predictions == 0) & (y == 1))
    
    return TP, FP, TN, FN

# accuracy function for the logistic regression
def calculate_accuracy(TP, TN, FP, FN):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy

# precision function for the logistic regression
def calculate_precision(TP, FP):
    precision = TP / (TP + FP)
    return precision

# relative path to the data file, assuming it's on the desktop
user_desktop = os.path.join(os.path.expanduser("~"), "Desktop")
# set the file path for the original file
file_path = os.path.join(user_desktop, "iris.data")

#-------------------------------------------
# LOADING AND SPLITTING THE DATA PORTION
#-------------------------------------------

# Load in the data
data = persistent_load(file_path)

# Split the dataset into training and validation sets (80% training, 20% validation) with shuffling
training_set, validation_set = train_test_split(data, test_size=0.2, random_state=42)

# Extract x features into an array
training_set_x = np.array([datapoint[:-1] for datapoint in training_set])

# Extract y feature into an array
training_set_y = np.array([int(datapoint[-1] == 'Iris-setosa') for datapoint in training_set])

# Extract x features into an array
validation_set_x = np.array([datapoint[:-1] for datapoint in validation_set])

# Extract y feature into an array and convert to integers
validation_set_y = np.array([int(datapoint[-1] == 'Iris-setosa') for datapoint in validation_set])

# Adding an intercept
training_set_x = np.insert(training_set_x, 0, 1, axis=1)  # Add intercept term
validation_set_x = np.insert(validation_set_x, 0, 1, axis=1)

# Initial theta values
initial_thetas = np.zeros(training_set_x.shape[1])  # Shape[1] gives the number of columns (features)

# Use minimize function to optimize parameters
result = minimize(cost_function, initial_thetas, args=(training_set_x, training_set_y), jac=gradient, method='BFGS')

# Retrieve the optimized parameters
optimal_thetas = result.x

# Predict the class labels using the estimated parameters
predictions = predict(validation_set_x, optimal_thetas)

# Get the confusion matrix
TP, TN, FP, FN = confusion_matrix(predictions, validation_set_y)

# accuracy
accuracy = calculate_accuracy(TP, TN, FP, FN)

# precision
precision = calculate_precision(TP, FP)

# Print all the results out
print("Optimal thetas:", optimal_thetas)
print("True Positive (TP): ", TP)
print("False Positive (FP): ", FP)
print("True Negative (TN): ", TN)
print("False Negative (FN): ", FN)
print("Accuracy: ", accuracy)
print("Precision: ", precision)

