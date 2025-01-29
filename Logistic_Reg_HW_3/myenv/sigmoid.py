#-------------------------------------------------------------------------
# HW03: Logistic Regression For Iris Dataset
# CS430-01 Spring 2024
# Authors: Sabrina and Megan Jenckes
# Date: March 8th, 2024
# IDE: Visual Studio Code version 1.87.1 February 2024
#
# File: sigmoid.py
#
# Purpose: Implement and test logistic regression by using the sigmoid function found in
# sigmoid.py (this file) to find the hypothesis.
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# Libraries needed to complete the program:
#-------------------------------------------------------------------------
import numpy as np

#-------------------------------------------------------------------------
# Functions:
#--------------------------------------------------------------------------
def sigmoid(z):
    """
    Calculates the sigmoid (works with matrices as well)
    """
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid
