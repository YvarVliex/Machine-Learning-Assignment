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
X_red = X.iloc[:,2:4].to_numpy()

#%%
# Apply the elbow method to determine good amount of clusters

max_clusters = 12
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
clusters =4

# Apply KMeans on the relevant columns of data (lat and lon of pickup)
kmeans = KMeans(n_clusters=clusters).fit(X_red)

X_ass = [X_red, np.transpose(kmeans.labels_)]
centroids = pd.DataFrame(kmeans.cluster_centers_, columns = ['xloc', 'yloc'])

#%%
#Plot the final clusters and their centroids

#colors = ['red', 'blue', 'green', 'purple', 'orange']
colors = plt.cm.get_cmap('hsv', clusters)
colors2 = [mcolors.rgb2hex(colors(i)) for i in range(colors.N)]

y_kmeans = kmeans.labels_
x_column = 0
y_column = 1


#plot taxi districts
for shape, center in zip(taxi_zones_shape_data.shapeRecords(), center_points):
	x = [i[0] for i in shape.shape.points[:]]
	y = [i[1] for i in shape.shape.points[:]]
	plt.plot(x, y, c = 'k', linewidth=0.3, zorder = 1)

#plot each pick up point with a color corresponding to its cluster
for i in range(clusters):    
    plt.scatter(X_red[y_kmeans == i, x_column], X_red[y_kmeans == i,y_column],s=20,c=colors2[i])

#plot all cluster centers in yellow/black
plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1],  marker = 'o', s=80,
           c = centroids.index.map(lambda x: 'k'), zorder = 2)
plt.scatter(centroids.iloc[:,0], centroids.iloc[:,1],  marker = 'o', s=50,
           c = centroids.index.map(lambda x: 'yellow'), zorder = 3)
plt.title('Optimal Locations for Dispatch Centers')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.show()