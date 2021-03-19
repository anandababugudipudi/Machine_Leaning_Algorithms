# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 20:28:41 2021

@author: anand
"""

# Importing necessary packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets.samples_generator import make_blobs

# Crate 40 Separable points
X, y = make_blobs(n_samples = 40, centers = 2, random_state = 20)

# Create a SVM Model instance and fit the data
model = svm.SVC(kernel = 'linear', C = 1).fit(X,y)

# Display the data in graph form
plt.scatter(X[:, 0], X[:, 1], c = y, s = 30, cmap = plt.cm.Paired)
# plt.show()

# # Checking the new data
# newData = [[3,4], [5,6]]
# print(model.predict(newData))

# Plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate a model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
ax.contour(XX, YY, Z, colors = 'k', levels = [-1, 0, 1], alpha = 0.5, linestyles = ['--', '-', '--'])

# Plot support vectors
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s = 100, linewidth = 1, facecolors = 'none')
plt.show()