# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:14:57 2021

@author: anand
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
data = dataset.copy(deep = True)

# Selecting the required data
X = data.iloc[:, [2 ,3]].values
y = data.iloc[:, 4].values

# Splitting the dataset into train data and test data
from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
XTrain = sc.fit_transform(XTrain)
XTest = sc.transform(XTest)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(XTrain,yTrain)

# Predicting the test results
yPred = classifier.predict(XTest)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yTest, yPred)

# Visualising the Training Set results
from matplotlib.colors import ListedColormap
XSet, ySet, = XTrain, yTrain
X1, X2 = np.meshgrid(np.arange(start = XSet[:, 0].min() - 1, stop = XSet[:, 0].max() + 1, step = 0.01),
                     np.arange(start = XSet[:, 1].min() - 1, stop = XSet[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(ySet)):
    plt.scatter(XSet[ySet == j, 0], XSet[ySet == j, 1], 
                cmap = ListedColormap(('red', 'green')), label = j)
plt.title('Naive Bayes (Training Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test Set results
from matplotlib.colors import ListedColormap
XSet, ySet, = XTest, yTest
X1, X2 = np.meshgrid(np.arange(start = XSet[:, 0].min() - 1, stop = XSet[:, 0].max() + 1, step = 0.01),
                     np.arange(start = XSet[:, 1].min() - 1, stop = XSet[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(ySet)):
    plt.scatter(XSet[ySet == j, 0], XSet[ySet == j, 1], 
                cmap = ListedColormap(('red', 'green')), label = j)
plt.title('Naive Bayes (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

from sklearn import metrics
mcr = metrics.classification_report(yTest, yPred)
print(mcr)








