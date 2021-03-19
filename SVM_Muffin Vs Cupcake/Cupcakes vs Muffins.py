# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 12:40:31 2021

@author: anand
"""
# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale = 1.2)

# Loading the datafile
recipes = pd.read_csv('recipes_muffins_cupcakes.csv')
# Plotting the muffins and cakes based on Flour and Sugar
sns.lmplot('Flour', 'Sugar', data = recipes, hue = 'Type', palette = 'Set1', fit_reg = False, scatter_kws = {"s" : 70})

# Pre-processing the data
type_label = np.where(recipes['Type'] == 'Muffin', 0, 1)
recipe_features = recipes.columns.values[1:].tolist()
ingredients = recipes[['Flour', 'Sugar']].values

# Model Training
model = svm.SVC(kernel = 'linear').fit(ingredients, type_label)

# Getting separating hyperplane
w = model.coef_[0]
a = -w[0] / w[1]
x = np.linspace(30, 60)
y = a * x -(model.intercept_[0]) / w[1]

# Getting the Parellels to the separating hyperplane
b = model.support_vectors_[0]
y_down = a * x + (b[1] - a * b[0])

c = model.support_vectors_[1]
y_up = a * x + (c[1] - a * c[0])

# Plotting the planes
sns.lmplot('Flour', 'Sugar', data = recipes, hue = 'Type', palette = 'Set1', fit_reg = False, scatter_kws= {"s": 70})
plt.title('Flour Vs Sugar')
plt.plot(x, y, linewidth = 2, color = 'black')
plt.plot(x, y_down, 'k--')
plt.plot(x, y_up, 'k--')

# Create a function to predict muffin or cupcake
def muffin_or_cupcake(flour, sugar):
    if (model.predict([[flour, sugar]]) == 0):
        print('Muffin Recipe.')
    else:
        print('Cupcake Recipe.')

# Predict if 50 parts flour and 20 parts sugar
muffin_or_cupcake(50, 20)

# Plot this point on graph
sns.lmplot('Flour', 'Sugar', data = recipes, hue = 'Type', palette = 'Set1', fit_reg = False, scatter_kws= {"s": 70})
plt.title('50 parts flour and 20 parts sugar')
plt.plot(x, y, linewidth = 2, color = 'black')
plt.plot(50, 20, 'yo', markersize = 9)