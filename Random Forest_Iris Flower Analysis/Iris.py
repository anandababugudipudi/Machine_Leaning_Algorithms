# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 15:15:29 2021

@author: anand
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataFrame = pd.read_csv('Iris.csv', index_col = False)
data = dataFrame.copy(deep = True)
data = data.drop(['Id'], axis = 1)

cleanupObjects = {'Species' : {'Iris-setosa' : 1,
                               'Iris-versicolor' : 2,
                               'Iris-virginica' : 3}}
data = data.replace(cleanupObjects)

X = data.values[:, 0:4]
y = data.values[:, 4]

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.25, random_state = 1)

clf = RandomForestClassifier(n_jobs = 2, random_state = 1).fit(XTrain, yTrain)

yPred = clf.predict(XTest)

accuracy = accuracy_score(yTest, yPred) * 100

print(f"The accuracy of the model is {round(accuracy, 2)} %")

# Testing an instance
predNow = clf.predict([[4.9, 3, 1.4, 0.2]])
print(predNow)