#-------------------------------------------------------------------------
# HW04: K-Means Clustering on an Example Dataset
# CS430-01 Spring 2024
# Authors: Sabrina and Megan Jenckes
# Date: April 20th, 2024
# IDE: Visual Studio Code version 1.86.0 January 2024
#
# File: _main_.py
#
# Purpose: Implement the K-Means Clustering algorithm on an example dataset, along with a visualization of the clusters.
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# Libraries needed to complete the program:
#-------------------------------------------------------------------------
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.io as sio

#-------------------------------------------------------------------------
# Functions:
#--------------------------------------------------------------------------

# relative path to the data file, assuming it's on the desktop
user_desktop = os.path.join(os.path.expanduser("~"), "Desktop")
# set the file path for the original file
file_path = os.path.join(user_desktop, "K-means-data.mat")
# load the data from the file
data = sio.loadmat(file_path)

#-------------------------------------------
# LOADING CLUSTERING AND VISUALIZING THE DATA
#-------------------------------------------

# push the data into a pandas dataframe
df = pd.DataFrame(data['X'], columns=['x', 'y'])

# this will be used to store the settings for the KMeans model
initializations = ['k-means++', 'random']

# since we will be clustering in a loop to try different settings, we will store the data in a dictionary
results = {}

# for every desired initialization, we will cluster the data and store the appropiate data in the results dictionary
for i in range(0, len(initializations)):
    print(f"Current Initialization: {initializations[i]}")
    
    # create the KMeans model
    kmeans = KMeans(n_clusters=3, init=initializations[i], random_state=0)
    # fit the model to the data
    kmeans.fit(df)
    
    # store the x and y values in the results dictionary
    DATA = results[initializations[i] + '_data'] = df
    # store the labels in the results dictionary underneath a data key
    LABELS = results[initializations[i] + '_labels'] = kmeans.labels_
    # store the cluster centers in the results dictionary
    CENTROIDS = results[initializations[i] + '_centers'] = kmeans.cluster_centers_
    
    # output the data, labels, and centroids to the console
    print('Centroids:', CENTROIDS)
    print('Labels:', LABELS)
    print('Data:', DATA.head(), '\n')
    
    #-------------------------------------------
    # VISUALIZING THE CLUSTERS
    #-------------------------------------------
    
    # to Megan: this is where you come in! you should be able to visualize the data using this loop using the provided variables (DATA, LABELS, CENTROIDS) above! thanks for your help! :3
