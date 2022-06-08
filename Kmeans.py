# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 08:54:05 2022

@author: cavaw
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import random as rd
import seaborn as sns
from math import ceil
from sklearn.cluster import KMeans
from read_data import load_preprocessed_data, plot_taxi_zones, read_taxi_zones_shape_file, determine_centers_of_districts

dataset = np.load('processed_trip_data_1.npy')

taxi_zones_shape_data = read_taxi_zones_shape_file('taxi_zones_shape_files/taxi_zones.shp')

center_points = determine_centers_of_districts(taxi_zones_shape_data)


X = pd.DataFrame(dataset, columns=['hour', 'day', 'xloc', 'yloc'])
#print(X)
X_red = X.iloc[:,2:4].to_numpy()
# print(X_red)

#%%
# Apply the elbow method to determine good amount of clusters

max_clusters = 5
# Decide on number of clusters
wcss = []
for i in range(1,max_clusters+1):
    print('trying', i, 'clusters now')
    kmeans = KMeans(n_clusters=i).fit(X_red)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,max_clusters+1),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('J')
plt.show()


#%%

# number of clusters
clusters = 5

# Apply KMeans on the relevant columns of data (lat and lon)
kmeans = KMeans(n_clusters=clusters).fit(X_red)

X_ass = [X_red, kmeans.labels_]
centroids = pd.DataFrame(kmeans.cluster_centers_, columns = ['xloc', 'yloc'])
#%%
#Plot the final clusters and their centroids 
# colors = {0:'red', 1:'blue', 2:'green', 3:'yellow'}
colors = plt.cm.get_cmap('hsv', clusters)
colors2 = [mcolors.rgb2hex(colors(i)) for i in range(colors.N)]
print(colors2)
#plt.figure()

for shape, center in zip(taxi_zones_shape_data.shapeRecords(), center_points):
	x = [i[0] for i in shape.shape.points[:]]
	y = [i[1] for i in shape.shape.points[:]]
	plt.plot(x, y, c = 'k', linewidth=0.5, zorder = 1)
    
plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1],  marker = 'o', s=70,
           c = centroids.index.map(lambda x: 'red'), zorder = 2)
plt.title('Optimal Locations for Dispatch Centers')
plt.show()