# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 01:28:00 2021

@author: anand
"""
# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the data from file
dataset = pd.read_csv('Breast Cancer Detection Test.csv', index_col = 0, na_values = ['?', '??', "###"])
data = dataset.copy(deep = True)

'''
sns.jointplot('radius_mean', 'texture_mean', data = data)

sns.heatmap(data.corr())
'''
# Chicking for missing values
missing = data.isnull().sum()
## No missing values

# Preparing our data
X = data[['radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']]
y = data['diagnosis']

# Splitting data into train and test
from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.3, random_state = 101)
# Loading the log regression
from sklearn.linear_model import LogisticRegression
logModel = LogisticRegression().fit(XTrain, yTrain)
# Predicting the values
yPredict = logModel.predict(XTest)

# Checking the accuracy
from sklearn.metrics import classification_report
print(classification_report(yTest, yPredict))
