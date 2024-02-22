# Multivariate Linear Regression For Boston Housing Data
# AUTHORS: Garrett Thrower and Megan Jenckes
# Last Modified: 2/17/2024

# This script calculates the coefficients for a multivariate linear regression model.
# All done manually, since the point is to understand the math behind it.

import os
import numpy as np
import matplotlib.pyplot as plt

'''
Important data information. Each data point in the data set is ordered from left to right as follows:
 CRIM     per capita crime rate by town
 ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
 INDUS    proportion of non-retail business acres per town
 CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
 NOX      nitric oxides concentration (parts per 10 million)
 RM       average number of rooms per dwelling
 AGE      proportion of owner-occupied units built prior to 1940
 DIS      weighted distances to five Boston employment centres
 RAD      index of accessibility to radial highways
 TAX      full-value property-tax rate per $10,000
 PTRATIO  pupil-teacher ratio by town
 B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
 LSTAT    % lower status of the population
 MEDV     Median value of owner-occupied homes in $1000's
'''
# IMPORTANT FOR ABOVE: THE ABOVE INFORMATION WAS COPY-PASTED FROM THE DATASET. IN ORDER FOR THE BELOW CODE TO WORK, THE ABOVE STRING MUST BE REMOVED FROM THE DATASET TEXT FILE!!!
    
# we assume that every column relates back to the data types from before, so we can hardcode the column names.
column_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]


import os

def preprocess_data(file_path):
    """
    Reads a text file, removes newline characters from every odd line
    (considering human counting, where lines start at 1), and saves the
    modified content to a new file with '_cleaned' added to the original filename.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # process each line
    new_lines = []
    for i, line in enumerate(lines):
        if i % 2 == 0:  # we target odd-numbered lines, using human counting
            new_lines.append(line.strip())  # remove newline characters and trailing spaces
        else:
            new_lines.append(line)  # even-numbered lines are added as is
    
    # generate cleaned filename
    filename = os.path.basename(file_path)
    file_root, file_ext = os.path.splitext(filename)
    cleaned_filename = f"{file_root}_cleaned{file_ext}"
    cleaned_path = os.path.join(os.path.dirname(file_path), cleaned_filename)
    
    # write the processed lines back to a new file
    with open(cleaned_path, 'w') as file:
        file.writelines(new_lines)
    
    print("Data has been cleaned and saved to", cleaned_path)
    return cleaned_path

    
    
# relative path to the data file, assuming it's on the desktop.
user_desktop = os.path.join(os.path.expanduser("~"), "Desktop")

# we set two possible file paths, one for the original file and one for the cleaned file.
file_path = os.path.join(user_desktop, "boston.txt")
cleaned_file_path = os.path.join(user_desktop, "boston_cleaned.txt")


# check if the cleaned file exists, if it does, use it. If not, check if the original file exists, and if it does, clean it.
if os.path.exists(cleaned_file_path):
    print("Cleaned data file found, loading...")
    cleaned_dataset = cleaned_file_path
else:
    if os.path.exists(file_path):
        print("Unformatted data file found. Cleaning data...")
        cleaned_dataset = preprocess_data(file_path)
    else:
        print("Neither data file was found.")
        

# at this point, the file path to the cleaned data should be stored in the variable cleaned_dataset.
# we can now load the data from the file and begin processing it.

def persistent_load(file_path):
    """
    Loads a dataset line by line from a text file into a list of lists, where each entry is a string that relates to the data type.
    Once the algorithm hits a newline, it will assume that the next line is the start of a new data sample.
    """
    # open the file and read the lines
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # for every line, split the line by whitespace and add it to the data list
    data = []
    for line in lines:
        data.append(line.strip().split())
    
    return data

# I felt like the ability to grab all the data from a certain column would be useful, so I wrote a function for that.
def grab_data(dataDict, column_name):
    """
    Gets a certain type of data from the dataset, given the column name.
    """
    if column_name not in column_names:
        print(f"Column name {column_name} not found.")
        return []
    
    index = column_names.index(column_name)
    column_data = [float(sample[index]) for sample in dataDict]
    return column_data

# QOL FUNCTIONS #

def predict(theta_values, features):
    """
    Predicts the MEDV value for a given set of features.

    Parameters:
    - theta_values: The theta values obtained from the gradient descent function.
    - features: The feature values for the new data point. This should be a 1D array.

    Returns:
    - The predicted MEDV value.
    """
    return np.dot(theta_values, features)

def normalize_data(data):
    """
    Normalizes the data by subtracting the mean and dividing by the standard deviation.

    Parameters:
    - data: The data to be normalized.

    Returns:
    - The normalized data.
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    return (data - mean) / std_dev

def unnormalize_data(normalized_value, mean, std_dev):
    """
    Unnormalizes a single value using the specified mean and standard deviation.

    Parameters:
    - normalized_value: The normalized data value.
    - mean: The mean used for normalization.
    - std_dev: The standard deviation used for normalization.

    Returns:
    - The unnormalized data value.
    """
    return (normalized_value * std_dev) + mean

def compare_predictions(training_features, training_target, validation_features, validation_target, final_thetas):
    """
    Compares the predicted values to the actual values using the validation set.

    Parameters:
    - training_features: The features used for training the model.
    - training_target: The target values used for training the model.
    - validation_features: The features used for validation.
    - validation_target: The target values used for validation.
    - final_thetas: The theta values obtained from the gradient descent function.

    Returns:
    - A list of tuples, where each tuple contains the predicted and actual values for a given data point.
    
    Usage: 
    compare_predictions(training_features, training_target, validation_features, validation_target, final_thetas) where the training and validation features and targets are numpy arrays and final_thetas is a list of theta values.
    """

# calculate MSE
def calculateMSE(x, y, theta0, theta1, theta2):
    # find m, the number of training data sets
    m = len(y)
    
    # find the hypothesis or h0(x^i)
    hypothesis = theta0 + theta1 * x[:,1] + theta2 * x[:,2]
    
    # find the derivative by subtracting
    # y^i with the hypothesis
    derivative = hypothesis - y
    
    # find the sum of the squares of the derivative
    sumOfSquares = np.sum(derivative ** 2)
    
    # find the mean squared error
    mse = (1/(2*m)) * sumOfSquares
    
    # return the mean squared error
    return mse

# Gradient Descent
def batchGradientDescent(X, y, thetas, alpha, goalAccuracy):
    m = len(y)
    while True:
        hypothesis = X.dot(thetas)
        errors = hypothesis - y
        gradient = (1/m) * X.T.dot(errors)
        new_thetas = thetas - alpha * gradient
        accuracy = np.sum(np.abs(new_thetas - thetas))
        
        if accuracy <= goalAccuracy:
            break
        
        thetas = new_thetas
    
    return thetas

# load data 
data_dict = persistent_load(cleaned_dataset)

# since we have the hard-coded column names, we can just use the index of the column name to get the data from the dataset.
# ill print out the first 5 data points for each column to show that the data was loaded correctly.
for column_name in column_names:
    print(column_name, ":", grab_data(data_dict, column_name)[:5])
print("\n\n")
    
# splitting the data into training and validation sets
# Extract the last 50 rows
validation_set = data_dict[-50:]

# Separate the rest of the data
training_set = data_dict[:-50]

# ( THIS WAS JUST FOR TESTING...)
# # Output or process the separated data
# print("Rest of the data:")
# for line in training_set:
#     print(str(line).strip())

# print("\nLast 50 rows:")
# for line in validation_set:
#     print(str(line).strip())


# for column_name in column_names:
#     print(column_name, ":", grab_data(validation_set, column_name)[:5])
# print("\n\n")

# normalizing training set (maybe working?? THIS NEEDS NUMPY)
numeric_training_set = np.array(training_set, dtype=float)

training_mean = np.mean(numeric_training_set)
training_std_dev = np.std(numeric_training_set)

normalized_training_setNP = (numeric_training_set - training_mean) / training_std_dev

# I converted the normalized training set back to a list just in case
normalized_training_set = normalized_training_setNP.tolist()

# normalizing validation set (maybe working?? THIS NEEDS NUMPY)
numeric_validation_set = np.array(validation_set, dtype=float)

validation_mean = np.mean(numeric_validation_set)
validation_std_dev = np.std(numeric_validation_set)

normalized_validation_setNP = (numeric_validation_set - validation_mean) / validation_std_dev

# I converted the normalized validation set back to a list just in case
normalized_validation_set = normalized_validation_setNP.tolist()

# Initial alpha / learning rate 
alpha = 0.01          # the homework's initial value

# Initial stopping accuracy
goalAccuracy = 0.01   # the homework's initial value

# (THIS IS FOR part a...)

# AGE data
age_data = np.array(grab_data(normalized_training_set, "AGE"))

# TAX data
tax_data = np.array(grab_data(normalized_training_set, "TAX"))

# MEDV data 
medv_data = np.array(grab_data(normalized_training_set, "MEDV"))

# Setting up x (the features including an intercept of 1 for x0, x1 is the age, and x2 is the tax)
x = np.column_stack((np.ones_like(age_data), age_data, tax_data))

# Initial theta values
initial_thetas = np.zeros(3)


# Call the function in order to get the final 
# results for the theta parameters
final_thetas = batchGradientDescent(x, medv_data, initial_thetas, alpha, goalAccuracy)


# Print the final theta parameters to the screen
print("Final thetas:", final_thetas)

# use the final thetas to calculate the mean squared error
mse = calculateMSE(x, medv_data, final_thetas[0], final_thetas[1], final_thetas[2])
print("Mean Squared Error:", mse)


