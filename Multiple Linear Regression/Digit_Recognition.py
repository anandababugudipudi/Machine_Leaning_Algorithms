# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 18:32:20 2021

@author: anand
"""

# 1. Importing the Libraries
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Loading the dataset
digits = load_digits()

# 2.1. Selecting the independant and dependant variables
X = digits.data
y = digits.target


# # 3. Displaying some of the digits
# plt.figure(figsize = (20, 4))
# for index, (image, label) in enumerate(zip(digits.data[:5], digits.target[:5])):
#     plt.subplot(1, 5, index + 1)
#     plt.imshow(np.reshape(image, (8, 8)), cmap = plt.cm.gray)
#     plt.title(f'Training: {label}\n', fontsize = 20)
## Displayed the digits images from 0 to 4
   
# 4. Dividing the dataset into Training and Test set
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = 0.25, random_state = 0)

# 5 Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
XTrain = sc.fit_transform(XTrain)
XTest = sc.transform(XTest) 

# 6. Training on our Logistic Regression model
from sklearn.linear_model import LogisticRegression
logisticReg = LogisticRegression().fit(XTrain, yTrain)

# 7.1 Prediciting the output of the first element in the test set
yPred1 = logisticReg.predict(XTest[0].reshape(1, -1))
## Predicted the digit as "2"

# 7.2 Predicting the output of first 10 elements of the test set
yPred1to10 = logisticReg.predict(XTest[:10])
## The output is [2 8 2 6 6 7 1 9 8 5]

# 8. Predicting the entire dataset
yPred = logisticReg.predict(XTest)
# print(yPred)
## Predicts the whole dataset

# 8.1 Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yTest, yPred)

# 9. Determing the accuracy of the model
accuracyScore = logisticReg.score(XTest, yTest)
accuracyr2Score = r2_score(yTest, yPred)
# print(accuracyScore)   # => 0.9666666666666667
# print(accuracyr2Score) # => 0.8721387273283021

# 10. Representing the confusion matrix in a heat map
plt.figure(figsize = (9, 9))
sns.heatmap(cm, annot = True, fmt = ".2f", linewidths = .5, square = True, cmap = 'RdYlGn_r')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title(f'Accuracy Score {round(accuracyScore,4)}', size = 15)
