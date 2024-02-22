#-------------------------------------------------------------------------
# HW02: Multivariate Linear Regression For Boston Housing Data
# CS430-01 Spring 2024
# Authors: Garrett Thrower and Megan Jenckes
# Date: February 22nd, 2024
# IDE: Visual Studio Code version 1.86.0 January 2024
#
# File: __main__.py
#
# Purpose: Implement and test multivariable gradient descent
# to find the parameters for linear regression using the dataset from the
# boston.txt file. 
#-------------------------------------------------------------------------

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
    
#-------------------------------------------------------------------------
# Libraries needed to complete the program:
#-------------------------------------------------------------------------
import os
import numpy as np

#-------------------------------------------------------------------------
# Functions:
#--------------------------------------------------------------------------
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

def normalize_data(data):
    """
    Normalizes the data by subtracting the mean and dividing by the standard deviation.

    Parameters:
    - data: The data to be normalized.

    Returns:
    - The normalized data.
    """
    numeric_set = np.array(data, dtype=float)
    mean = np.mean(numeric_set)
    std_dev = np.std(numeric_set)
    temp_np_array = (numeric_set - mean) / std_dev

    return temp_np_array.tolist()

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

# Gradient Descent
def batch_gradient_descent(x, y, thetas, alpha, goal_accuracy):
    m = len(y)
    while True:
        hypothesis = x.dot(thetas)
        errors = hypothesis - y
        gradient = (1/m) * x.T.dot(errors)
        new_thetas = thetas - alpha * gradient
        accuracy = np.sum(np.abs(new_thetas - thetas))
        
        if accuracy <= goal_accuracy:
            break
        
        thetas = new_thetas
    
    return thetas

def calculate_MSE(x, y, thetas):
    predictions = x.dot(thetas)
    errors = predictions - y
    mse = (1/(2*len(y))) * np.sum(errors ** 2)
    return mse

# Normal Equation (for part a ONLY)
def normal_equation(x, y):
    """
    Calculates the thetas to predict MEDV using AGE and TAX

    Parameters:
    - x: The array that contains the AGE and TAX data from the normalized training set
    - y: The MEDV from the normalized training set. This should be a 1D array.

    Returns:
    - The thetas needed to predict MEDV.
    """
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)


#-------------------------------------------------------------------------
# Variables / loading and processing the data:
#-------------------------------------------------------------------------

# we assume that every column relates back to the data types from before, so we can hardcode the column names.
column_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

# Initial alpha / learning rate 
alpha = 0.01          # the homework's initial value

# Initial stopping accuracy
goal_accuracy = 0.01   # the homework's initial value

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

# load data 
data_dict = persistent_load(cleaned_dataset)
    
# splitting the data into training and validation sets
# Extract the last 50 rows
validation_set = data_dict[-50:]

# Separate the rest of the data
training_set = data_dict[:-50]

# # normalizing training set
normalized_training_set = normalize_data(training_set)

# normalizing validation set 
normalized_validation_set = normalize_data(validation_set)

#-------------------------------------------------------------------------
# PART ONE A:
#-------------------------------------------------------------------------

# AGE data
age_data = np.array(grab_data(normalized_training_set, "AGE"))

# TAX data
tax_data = np.array(grab_data(normalized_training_set, "TAX"))

# MEDV data 
medv_data = np.array(grab_data(normalized_training_set, "MEDV"))

# Setting up x (the features including an intercept of 1 for x0, x1 is the age, and x2 is the tax)
x_2a = np.column_stack((np.ones_like(age_data), age_data, tax_data))

# Initial theta values
initial_thetas_2a = np.zeros(3)

# Call the function in order to get the final 
# results for the theta parameters
final_thetas_2a = batch_gradient_descent(x_2a, medv_data, initial_thetas_2a, alpha, goal_accuracy)

# AGE data
age_data_val = np.array(grab_data(normalized_validation_set, "AGE"))

# TAX data
tax_data_val = np.array(grab_data(normalized_validation_set, "TAX"))

# MEDV data 
medv_data_val = np.array(grab_data(normalized_validation_set, "MEDV"))

# Setting up x (the features including an intercept of 1 for x0, x1 is the age, and x2 is the tax)
x_2a_val = np.column_stack((np.ones_like(age_data_val), age_data_val, tax_data_val))

# use the final thetas to calculate the mean squared error
mse_2a = calculate_MSE(x_2a_val, medv_data_val, final_thetas_2a)


#-------------------------------------------------------------------------
# PART ONE B:
#-------------------------------------------------------------------------

# in part B, we will be factoring in all the features of the dataset.

crim_data = np.array(grab_data(normalized_training_set, "CRIM"))
zn_data = np.array(grab_data(normalized_training_set, "ZN"))
indus_data = np.array(grab_data(normalized_training_set, "INDUS"))
chas_data = np.array(grab_data(normalized_training_set, "CHAS"))
nox_data = np.array(grab_data(normalized_training_set, "NOX"))
rm_data = np.array(grab_data(normalized_training_set, "RM"))
dis_data = np.array(grab_data(normalized_training_set, "DIS"))
rad_data = np.array(grab_data(normalized_training_set, "RAD"))
ptratio_data = np.array(grab_data(normalized_training_set, "PTRATIO"))
b_data = np.array(grab_data(normalized_training_set, "B"))
lstat_data = np.array(grab_data(normalized_training_set, "LSTAT"))

# Setting up x (the features including an intercept of 1 for x0, x1 is the crim, and x2 is the zn...etc)
x_2b = np.column_stack((np.ones_like(crim_data), crim_data, zn_data, indus_data, chas_data, nox_data, rm_data, age_data, dis_data, rad_data, tax_data, ptratio_data, b_data, lstat_data))

# Initial theta values
initial_thetas_test = np.zeros(14)

# Call the function in order to get the final 
# results for the theta parameters
final_thetas_2b = batch_gradient_descent(x_2b, medv_data, initial_thetas_test, alpha, goal_accuracy)

crim_data_val = np.array(grab_data(normalized_validation_set, "CRIM"))
zn_data_val = np.array(grab_data(normalized_validation_set, "ZN"))
indus_data_val = np.array(grab_data(normalized_validation_set, "INDUS"))
chas_data_val = np.array(grab_data(normalized_validation_set, "CHAS"))
nox_data_val = np.array(grab_data(normalized_validation_set, "NOX"))
rm_data_val = np.array(grab_data(normalized_validation_set, "RM"))
dis_data_val = np.array(grab_data(normalized_validation_set, "DIS"))
rad_data_val = np.array(grab_data(normalized_validation_set, "RAD"))
ptratio_data_val = np.array(grab_data(normalized_validation_set, "PTRATIO"))
b_data_val = np.array(grab_data(normalized_validation_set, "B"))
lstat_data_val = np.array(grab_data(normalized_validation_set, "LSTAT"))

# Setting up x (the features including an intercept of 1 for x0, x1 is the crim, and x2 is the zn...etc)
x_2b_val = np.column_stack((np.ones_like(crim_data_val), crim_data_val, zn_data_val, indus_data_val, chas_data_val, nox_data_val, rm_data_val, age_data_val, dis_data_val, rad_data_val, tax_data_val, ptratio_data_val, b_data_val, lstat_data_val))

# use the final thetas to calculate the mean squared error
mse_2b = calculate_MSE(x_2b_val, medv_data_val, final_thetas_2b)

#-------------------------------------------------------------------------
# PART TWO FOR PART A ONLY:
#-------------------------------------------------------------------------

# Call the function in order to get the final
# results for the theta parameters using the normal equation
final_thetas_norm = normal_equation(x_2a, medv_data)

# use the final thetas from the normal equation to calculate the mean squared error
mse_norm = calculate_MSE(x_2a, medv_data, final_thetas_norm)

#-------------------------------------------------------------------------
# OUTPUT:
#-------------------------------------------------------------------------

# Path to the output file on the desktop.
output_file_path = os.path.join(user_desktop, "output.txt")

# Create or open the output file for writing.
with open(output_file_path, "w") as output_file:
    # Outputting Part One: The theta values using gradient descent
    # and the mean squared error 
    output_file.write("Multivariate Linear Regression For Boston Housing Data\n\n")
    output_file.write("PART ONE: Gradient Descent and Mean Squared Error\n")
    output_file.write("2a: Predicting MEDV based on AGE and TAX\n")
    # Print the final theta parameters to the output file
    output_file.write(f"Final thetas of 2a: {final_thetas_2a}\n")
    output_file.write(f"Mean Squared Error of 2a: {mse_2a}\n\n")
    # Part B
    output_file.write("2b: Predicting MEDV based on all features\n")
    output_file.write(f"Final thetas of 2b: {final_thetas_2b}\n")
    output_file.write(f"Mean Squared Error of 2b: {mse_2b}\n\n")
    # Outputting Part Two: The theta values using the normal equation
    # and the mean squared error 
    output_file.write("PART TWO: Normal Equation and Mean Squared Error\n")
    output_file.write("2a: Predicting MEDV based on AGE and TAX\n")
    output_file.write(f"Final thetas of 2a using the normal equation: {final_thetas_norm}\n")
    output_file.write(f"Mean Squared Error of 2a using the normal equation: {mse_norm}\n")

print("All results has been printed to the output.txt file found on your desktop.")