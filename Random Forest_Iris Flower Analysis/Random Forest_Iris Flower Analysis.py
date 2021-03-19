# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 14:24:12 2021

@author: anand
"""

# Loading the library with the iris dataset
from sklearn.datasets import load_iris
 
# Loading the other packages
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Setting random seed
np.random.seed(0)

# Creating an object with iris data
iris = load_iris()

# Creating a DataFrame with the four feature variables
dataframe = pd.DataFrame(iris.data, columns = iris.feature_names)

# Creating a local copy of the data
data = dataframe.copy(deep = True)

# Adding a new column fro the species name
data['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Creating Test and Train Data
data['is_train'] = np.random.uniform(0, 1, len(data)) <= .75

# Creating dataframes with test rows and training rows
train, test = data[data['is_train'] == True], data[data['is_train'] == False]

# Create a list of feature column's names
features = data.columns[:4]

# Converting each species name into digits
y = pd.factorize(train['species'])[0]
      
# Creating a Random Forest Classifier
clf = RandomForestClassifier(n_jobs = 2, random_state = 0)
# Training the classifier
clf.fit(train[features], y)

# Predicting the outputs
yPred = clf.predict(test[features])

# Viewing the predicted probabilities of the first 10 observations
predProb = clf.predict_proba(test[features])

# Mapping the names for the plants for each predicted plant class
preds = iris.target_names[clf.predict(test[features])]

# Creating Confusion Matrix
confMatrix = pd.crosstab(test['species'], preds, rownames = ['Actual Names'], colnames = ['Predicted Species'])

# Checking for random input of features
predictNow = iris.target_names[clf.predict([[5.0, 3.6, 1.4, 2.0]])]
print(f"The given features belongs to '{predictNow[0]}' ")







