#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 20:13:02 2018

@author: martin
"""
import numpy as np

from sklearn.metrics import r2_score
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.impute import MissingIndicator

#Setting NaN values to 0
y = np.genfromtxt("y_train.csv", delimiter=",", skip_header=1,filling_values=0)
X = np.genfromtxt("X_train.csv", delimiter=",", filling_values=0,skip_header=1)
y = y[:,1:]
X = X[:,1:]



#Splitting set into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

#Look what is missing a lot of data
indicator = MissingIndicator(missing_values=0,features = 'all')
mask_all = indicator.fit_transform(X_train)
missing_cols = mask_all.sum(axis=0)
indexes = np.argwhere(missing_cols > len(missing_cols)/10)


#fit a relly bad model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
score = r2_score(y_test, pred)
#Result, shit score
print(score)