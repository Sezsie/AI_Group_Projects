import numpy as np

def sigmoid(z):
    """
    calculates the sigmoid (works with matrices as well as required...)
    """
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid
