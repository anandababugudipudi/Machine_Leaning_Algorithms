# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 23:09:30 2021

@author: anand
"""

# importing necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

# Loading the data file
data_load = pd.read_csv('loan_data.csv', sep = ',', header = 0)
data = data_load.copy(deep = True)

# Encoding categorical data by Find and Replace Method
cleanupObjects = {'purpose' : {'debt_consolidation' : 0, 'all_other' : 1,'credit_card' : 2,'home_improvement' : 3,'small_business' : 4,
'major_purchase' : 5,'educational' : 6}}
data = data.replace(cleanupObjects)

# Splitting into Dependant and Independant variables
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# Splitting the Dataset into Train and Test data
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.30, random_state = 0)

# Creating a model instance and training the model
clfEntropy = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, max_depth = 3, min_samples_leaf = 14)
clfEntropy.fit(XTrain, yTrain)

# Predictihg the output
yPred = clfEntropy.predict(XTest)

# Checking the accuracy_score
accuracy = accuracy_score(yTest, yPred) * 100
print(f"The model is working with accuracy of {round(accuracy,2)} %")











































