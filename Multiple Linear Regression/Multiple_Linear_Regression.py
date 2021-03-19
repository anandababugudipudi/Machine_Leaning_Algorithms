# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:11:30 2021

@author: anand
"""

# 1
# Import the libraries
import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2
# Import and load the data
dataset = pd.read_csv('1000_Companies.csv')
data = dataset.copy(deep = True)

# Extracting the Independant and Dependant Variables
X = data.iloc[:, :-1].values
y = data.iloc[:, 4].values

# print(data.corr()) # Dependancy of each value on another

# 3
# Data Visualisation
sns.heatmap(data.corr())

# 4
# Encoding categorical data
# As the State column contains City name as string, we have to convert it into
#  numerics wit this step
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])

# State column change
columnTransfer = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
X = columnTransfer.fit_transform(X)


# 5
# Aviod Dummy Variable Trap
X = X[:, 1:]

# 6
# Splitting the data into train and test data
from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 0)

# 7
# Fitting Multiple Linear Regression Model to Training set
from sklearn.linear_model import LinearRegression
linearReg = LinearRegression().fit(XTrain, yTrain)

# 8
# Predicting the Test set results
yPred = linearReg.predict(XTest)

# 9
# Calculating the Coeffecients and Intercept
print(linearReg.coef_)
print(linearReg.intercept_) 

# 10
# Evaluating the model
from sklearn.metrics import r2_score
r2Score = r2_score(yTest, yPred)
print(r2Score)


































