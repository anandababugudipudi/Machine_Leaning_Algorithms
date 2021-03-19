# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:38:13 2021

@author: anand
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale = 1.2)
import quantstats as qs

# Importing the data
dataset = pd.read_csv('cars.csv')

data = dataset.copy(deep = True)

# Extracting the data except brand
X = dataset[:-1]

# Checking for missing data
missing = dataset.isnull().sum()
## No missing values are found

# Converting to numeric data, Taking data excluding brand column
X = data[data.columns[:-1]]
X = X.apply(pd.to_numeric, errors = 'coerce').fillna(0).astype(int)
summary_X = X.describe()

# Using the Elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(0, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()    
## We can take k as 3 from the plot

# Applying k-means to the cars dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
X = X.as_matrix(columns = None)
## X is an array of data and y_kmeans is clusters 0, 1, 2

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], size = 100, c = 'red', label = 'Toyota')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], size = 100, c = 'red', label = 'Nissan')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], size = 100, c = 'red', label = 'Honda')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of car make')
plt.legend()
plt.show()













































































