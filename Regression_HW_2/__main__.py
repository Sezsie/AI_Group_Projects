# Multivariate Linear Regression For Boston Housing Data
# AUTHORS: Garrett Thrower and Megan Jenckes
# Last Modified: 2/17/2024

# This script calculates the coefficients for a multivariate linear regression model.
# All done manually, since the point is to understand the math behind it.

import os

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

# load data 
data_dict = persistent_load(cleaned_dataset)
crime_data = grab_data(data_dict, "CRIM")

# since we have the hard-coded column names, we can just use the index of the column name to get the data from the dataset.
# ill print out the first 5 data points for each column to show that the data was loaded correctly.
for column_name in column_names:
    print(column_name, ":", grab_data(data_dict, column_name)[:5])
print("\n\n")
    
# all done for now, will continue later.
    
    
    
    