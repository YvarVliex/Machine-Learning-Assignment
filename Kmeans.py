# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 08:54:05 2022

@author: cavaw
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
import seaborn as sns
from math import ceil
from sklearn.cluster import KMeans
from read_data import load_preprocessed_data

dataset = load_preprocessed_data()



# Decide on number of clusters
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i).fit(X_red)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('J')
plt.show()

# Apply KMeans on the relevant columns of data (lat and lon)
kmeans = KMeans(n_clusters=3).fit(X_red.iloc[:,:2])

X_red['centroid'] = kmeans.labels_
centroids = pd.DataFrame(kmeans.cluster_centers_, columns = ['hp', 'weightlbs'])

#Plot the final clusters and their centroids 
colors = {0:'red', 1:'blue', 2:'green'}

plt.scatter(X_red.iloc[:,0], X_red.iloc[:,1],  marker = 'o', c = X_red['centroid'].apply(lambda x: colors[x]), alpha = 0.5)
plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1],  marker = 'o', s=300, 
           c = centroids.index.map(lambda x: colors[x]))
plt.show()