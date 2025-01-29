#-------------------------------------------------------------------------
# HW03: Logistic Regression For Iris Dataset
# CS430-01 Spring 2024
# Authors: Sabrina and Megan Jenckes
# Date: March 8th, 2024
# IDE: Visual Studio Code version 1.87.1 February 2024
#
# File: _main_.py
#
# Purpose: Implement and test logistic regression by using the sigmoid function found in
# sigmoid.py to find the hypothesis, the cost function to find the cost, and the gradient
# function to find the gradient of the cost. All of these functions will then be used to 
# find the optimal parameters for logistic regression using the dataset from the
# iris.data file. Implement and find the confusion matrix by making predictions
# using the optimal parameters found by the Broyden-Fletcher-Goldfarb-Shanno 
# optimization algorithm from the scipy library and calculate the accuracy and precision. 
#-------------------------------------------------------------------------

""" 
IRIS DATASET INFORMATION: 
The names of the classes of the iris dataset are as follows:
    1. Iris-setosa
    2. Iris-versicolor
    3. Iris-virginica
"""

#-------------------------------------------------------------------------
# Libraries needed to complete the program:
#-------------------------------------------------------------------------

import os                                               # used to read the dataset file from the desktop
import numpy as np                                      # used for arrays and math functions

from scipy.optimize import minimize                     # optimization of the cost function by minimizing it
                                                        # using the Broyden-Fletcher-Goldfarb-Shanno 
                                                        # optimization algorithm

from sigmoid import sigmoid                             # used to get the hypothesis for the cost function
                                                        # (the sigmoid function is implemented in the sigmoid.py)

from sklearn.model_selection import train_test_split    # used to split and shuffle the data into the training and
                                                        # validation sets, keeping the shuffle the same each time to
                                                        # ensure the same results each time

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
    # initialize empty list to store data values
    data = []
    # process each line
    for line in lines:
        # skip empty lines (This was for the extra lines at the end of the file)
        if not line.strip():
            continue
    
        values = line.strip().split(',')                  # splits the five different features apart
        data_values = [float(val) for val in values[:4]]  # extracts the four x features as floats
        data_values.append(values[4])                     # add the y value as a string
        data.append(data_values)                          # add the five values to the data nested list as a list
    return data

# now that the data has been read in and splitted below,
# we can calculate the cost
def cost_function(thetas, x, y):
    """
    Cost function for logistic regression:
    Takes in the initial thetas (0), the x and y features from the
    training set and calculates the cost using the sigmoid function
    found in the sigmoid.py that was imported above. 
    """
    # number of training examples
    m = len(y)
    # calculate the hypothesis
    hypothesis = sigmoid(np.dot(x, thetas))
    # clip the hypothesis to avoid 0 and 1 (the optimization method below dislikes when log is exactly 0 or 1)
    hypothesis = np.clip(hypothesis, 1e-15, 1 - 1e-15)
    # calculate the cost
    cost = (1 / m) * np.sum(-y * np.log(hypothesis) - (1 - y) * np.log(1 - hypothesis))
    return cost

# now that we have the cost below,
# we can calculate the gradient
def gradient_function(thetas, x, y):
    """
    Gradient function for logistic regression:
    Takes in the initial thetas (0), the x and y features from the
    training set and calculates the gradient using the sigmoid function
    found in the sigmoid.py that was imported above. 
    """
    # number of training examples
    m = len(y)
    # calculate the hypothesis
    hypothesis = sigmoid(np.dot(x, thetas))
    # calculate the gradient
    gradient = (1 / m) * np.dot(x.T, (hypothesis - y))
    return gradient

# now that we have both the cost and gradient that
# allows us to find the optimal parameters for logistic regression,
# we can predict the target value, y, or Iris-setosa in this case
def predict_function(x, theta):
    """
    Predict the class label for each sample in x using the optimal thetas.
    Takes in the x features and the optimal thetas and calculates the prediction.
    Returns the predictions as 0 if incorrect or 1 as correct. 
    """
    # calculate the predicted probability
    probability = 1 / (1 + np.exp(-np.dot(x, theta)))  # P(y = 1| x; theta)
    # convert probability to binary predictions
    predictions = (probability >= 0.5).astype(int)
    return predictions

# now that we have the predictions that we used to create the 
# confusion matrix below, we can calculate the accuracy and precision
def calculate_accuracy(TP, TN, FP, FN):
    """
    Calculates the accuracy using the true positive, 
    true negative, false positive, and false negative values
    found below from the confusion matrix.
    """
    #
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy

def calculate_precision(TP, FP):
    """
    Calculates the precision using the true positive
    and false positive values found below from the confusion matrix.
    """
    precision = TP / (TP + FP)
    return precision

#------------------------------------------------------
# LOADING, SPLITTING THE DATA, INITIALIZING VARIABLES:
#------------------------------------------------------

# relative path to the data file, assuming it's on the desktop
user_desktop = os.path.join(os.path.expanduser("~"), "Desktop")
# set the file path for the original file
file_path = os.path.join(user_desktop, "iris.data")

# load in the data
data = persistent_load(file_path)

# split the dataset into training and validation sets (80% training, 20% validation) with shuffling
training_set, validation_set = train_test_split(data, test_size=0.2, random_state=42)

# extract x features into an array
training_set_x = np.array([datapoint[:-1] for datapoint in training_set])

# extract y feature into an array and convert to integers (our goal target is for the Iris-setosa)
training_set_y = np.array([int(datapoint[-1] == 'Iris-setosa') for datapoint in training_set])

# extract x features into an array
validation_set_x = np.array([datapoint[:-1] for datapoint in validation_set])

# extract y feature into an array and convert to integers (our goal target is for the Iris-setosa)
validation_set_y = np.array([int(datapoint[-1] == 'Iris-setosa') for datapoint in validation_set])

# adding an intercept
training_set_x = np.insert(training_set_x, 0, 1, axis=1)  
validation_set_x = np.insert(validation_set_x, 0, 1, axis=1)

# initial theta values
initial_thetas = np.zeros(training_set_x.shape[1])  # shape[1] gives the number of columns (features)

#---------------------------------------------------------
# PART TWO/THREE: The cost, gradient, and optimization
#---------------------------------------------------------

# minimize the cost function in order to optimize it and get the optimal thetas
# jac = the jacobian for the gradient, method for optimization = Broyden-Fletcher-Goldfarb-Shanno optimization algorithm
optimize_result = minimize(cost_function, initial_thetas, args=(training_set_x, training_set_y), jac=gradient_function, method='BFGS')

# retrieve the optimized parameters (optimize_result also contains the value of the cost function, which we don't need)
optimal_thetas = optimize_result.x

# predict the class labels using the optimized parameters
predictions = predict_function(validation_set_x, optimal_thetas)

#---------------------------------------------------------
# PART FOUR/FIVE: The confusion matrix, accuracy, and precision
#---------------------------------------------------------

# True Positive = if the prediction is 1 and the actual class is 1
TP = np.sum((predictions == 1) & (validation_set_y == 1))
# False Positive = if the prediction is 1 and the actual class is 0
FP = np.sum((predictions == 1) & (validation_set_y == 0))
# True Negative = if the prediction is 0 and the actual class is 0
TN = np.sum((predictions == 0) & (validation_set_y == 0))
# False Negative = if the prediction is 0 and the actual class is 1
FN = np.sum((predictions == 0) & (validation_set_y == 1))

# create a 2D array for more of a "matrix representation"
confusion_matrix = np.array([[TP, FP], [FN, TN]])

# calculate the accuracy using the confusion matrix
accuracy = calculate_accuracy(TP, TN, FP, FN)

# calculate the precision using the confusion matrix
precision = calculate_precision(TP, FP)

#---------------------------------------------------------
# PART SIX: The final results
#---------------------------------------------------------

# Print all the results out
print("-----------------------------------------")
print("PART ONE: The Optimal Thetas:")
print("-----------------------------------------")
print("Optimal Thetas:", optimal_thetas)
print("-----------------------------------------")
print("PART TWO: The Confusion Matrix:")
print("GOAL: The Iris-setosa:")
print("-----------------------------------------")
print(confusion_matrix)
print("True Positive (TP): ", TP)
print("False Positive (FP): ", FP)
print("True Negative (TN): ", TN)
print("False Negative (FN): ", FN)
print("-----------------------------------------")
print("PART THREE: The Accuracy and Precision:")
print("-----------------------------------------")
print("Accuracy: ", accuracy)
print("Precision: ", precision)
