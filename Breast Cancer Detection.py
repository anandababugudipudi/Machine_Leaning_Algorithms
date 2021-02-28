# =================================================================================== #
#                 Breast Cancer Detection with Logistic Regression                    #
# =================================================================================== #
# Breast Cancer is one of the most common cancers among women worldwide, representing #
# the majority of new cancer cases and cancer-related deaths according to global      #
# statistics, making it a significant public health problem in today’s society. So    #
# gaining insights about the data related to this problem can give us chance to       #
# predict whether a patient has Malignant or Benign Cancer.                           #
# =================================================================================== #

# Importing necessary libraries
import numpy as np
import sklearn.datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Getting the datasets
breastCancer = sklearn.datasets.load_breast_cancer()

X = breastCancer.data
Y = breastCancer.target
print(f"Breast Cancer Dataset dimensions: {X.shape}")

data = pd.DataFrame(breastCancer.data, columns = breastCancer.feature_names)
data['class'] = breastCancer.target
dataInfo = data.describe()
# 0 - Malignant
# 1 - Benign
print("")
print(f"{breastCancer.target_names[0]} Cases : {data['class'].value_counts()[0]}")
print(f"{breastCancer.target_names[1]} Cases    : {data['class'].value_counts()[1]}")

# =================================================================================== #
#                           Train and Test Data Splitting                             #
# =================================================================================== #
# ====================
# Run 1: Observations
# ====================
# from sklearn.model_selection import train_test_split
# XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size = 0.1)
# print(Y.mean(), YTrain.mean(), YTest.mean())
# After the above run, the mean of actual, train and test data has come dissimilar
# which are supposed to be similar
# Here we have got =>  actual -> 0.6274165202108963 
#                      Train  -> 0.625 
#                      Test   -> 0.6491228070175439
# So we have to include a parameter called stratify in train_test_split

# ====================
# Run 2: Observations  
# ====================
# XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size = 0.1, stratify = Y)
# In Run 2 we have got almost similar values for mean
# We have got =>  actual -> 0.6274165202108963 
#                 Train  -> 0.626953125 
#                 Test   -> 0.631578947368421

XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size = 0.1, stratify = Y, random_state = 1)

# test_size     -> to split the test data in proportion
# stratify      -> for correct distribution of data as of the original data
# random_state  -> specific split of data. each value of random_state splits the data differently

# =================================================================================== #
#                       Training of Logistic Regression Model                         #
# =================================================================================== #
#from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

# Training the model on training data
classifier.fit(XTrain, YTrain)

# =================================================================================== #
#                               Evolution of the Model                                #
# =================================================================================== #
# from sklearn.metrics import accuracy_score
# Prediction on Training Data
prediction_on_training_data = classifier.predict(XTrain)
accuracy_on_training_data = accuracy_score(YTrain, prediction_on_training_data)
print("\nAccuracy of Training Data : ", accuracy_on_training_data)

# Prediction on Testing Data
prediction_on_testing_data = classifier.predict(XTest)
accuracy_on_testing_data = accuracy_score(YTest, prediction_on_testing_data)
print("Accuracy Testing Data    : ", accuracy_on_testing_data)
print()
# ===========================
# Observations of Evoluation 
# ===========================
# Accuracy on test data is less than that of the training data
#       Accuracy of Training Data :  0.951171875
#       Accuracy Testing Data     :  0.9298245614035088
# Here onwards we have to test the accuracy of same data on different models to find
# which one is providing highest accuracy among all


# =================================================================================== #
#           Detecting whether the Patient has Benign or Malignant Cancer              #
# =================================================================================== #
# For this we have to download the dataset from Kaggle
# https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
# Copy the data rows of the file to a list here

# For time being copy on instance for testing from the file
input_data = [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,
              0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,
              17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]

# Change this input data list to numpy array
input_data_array = np.array(input_data)

# Reshape the array as we are predicting the output for one instance
input_data_reshaped = input_data_array.reshape(1, -1)

# Prediction (0 - Malignant and 1 - Benign)
prediction = classifier.predict(input_data_reshaped)
if (prediction[0] == 0):
    print("The Brest Cancer is Malignant")
else:
    print("The Brest Cancer is Benign")

# =============
# The output is 
# =============
# The Brest Cancer is Malignant

# Testing for Benign Cancer input data
# input_data = [8.888,14.64,58.79,244,0.09783,0.1531,0.08606,0.02872,0.1902,0.0898,0.5262,
#                0.8522,3.168,25.44,0.01721,0.09368,0.05671,0.01766,0.02541,0.02193,9.733,
#                15.67,62.56,284.4,0.1207,0.2436,0.1434,0.04786,0.2254,0.1084]
# Output is -> The Brest Cancer is Benign

# =================================================================================== #
# You can try: Predict this by using K-Nearest Neighbuors algorithm and compare the   #
# accuracy score for Logistic Regression and KNN                                      #
# =================================================================================== #
