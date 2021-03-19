# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 16:19:17 2021

@author: anand
"""
# =============================================================================
# KNN - Predict whether a person will get Diabetes or not
# =============================================================================
# Importing the necessary packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def chooseKNeighbours(yTest):
    import math
    k = int(math.sqrt(len(yTest)))
    if (k%2==0):
        return (k-1)
    else:
        return (k)

# Importing the datafile into DataFrame
dataFrame = pd.read_csv('diabetes.csv')
data = dataFrame.copy(deep = True)

# Replace Zeros
zeroNotAccepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
for col in zeroNotAccepted:
    data[col] = data[col].replace(0, np.NaN)
    meanHere = int(data[col].mean(skipna = True))
    data[col] = data[col].replace(np.NaN, meanHere)
    
# Splitting the dataset into features and label
X = data.values[:, 0:8]
y = data.values[:, 8]

# Splitting the data into Train and Test Data
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
ss = StandardScaler()
XTrain = ss.fit_transform(XTrain)
XTest = ss.transform(XTest)



# Creating a KNN Model instance and fitting the model
KNN = KNeighborsClassifier(n_neighbors = chooseKNeighbours(yTest), p = 2, metric = 'euclidean')
KNN.fit(XTrain, yTrain)
# p=2 as we are looking for diabetic or not

# Predicting the model output
yPred = KNN.predict(XTest)
# Evaluate the Model
confMatrix = confusion_matrix(yTest, yPred)
accuracy = accuracy_score(yTest, yPred)
f1Score = f1_score(yTest, yPred)

print("Confusion Matrix: ")
print(confMatrix)
print(f"Accuracy score is {round(accuracy,2)*100} %")
print(f"F1 score is {round(f1Score,2)*100} %")