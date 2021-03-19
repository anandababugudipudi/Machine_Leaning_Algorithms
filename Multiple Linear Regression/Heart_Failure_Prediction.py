# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 20:24:04 2021

@author: anand
"""

# 1. Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# 2. Importing the data from csv File
dataset = pd.read_csv('heart_failure_clinical_records_dataset.csv')
data = dataset.copy(deep = True)

# 3. Selecting the Independant and dependant data
X = data.iloc[:,: -1].values
y = data.iloc[:, 12].values

# 4. Splitting the data into Training and Test Set
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.20, random_state = 1)

# 5. Features scaling
ss = StandardScaler()
XTrain = ss.fit_transform(XTrain)
XTest = ss.transform(XTest)

# 6. Applying logistic regression on the train data
from sklearn.linear_model import LogisticRegression
logisticReg = LogisticRegression().fit(XTrain, yTrain)

# 7. Predicting the model accuracy
yPred = logisticReg.predict(XTest)

# 8. Accuracy measuring
accuracy = accuracy_score(yTest, yPred)
## The Output => 0.8833333333333333

cm = confusion_matrix(yTest, yPred)
## The Output 
 #               [[43  3]
 #                [ 4 10]]

# 9. Extracting TN, TP, FP, FN
tn, fp, fn, tp = confusion_matrix(yTest, yPred).ravel()

# 10. Confustion Matrix Metrics
matrix = classification_report(yTest, yPred)
print(f"Classification Report: {matrix}\n")
