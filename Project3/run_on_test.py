#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 16:25:52 2018

@author: martin
"""
from sklearn.feature_selection import mutual_info_regression, SelectPercentile
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import LabelEncoder
import csv


print('------ opening files -------')
X = np.loadtxt("X.txt", delimiter=",", dtype="float64")
X_test = np.loadtxt("X_test.txt", delimiter=",", dtype="float64")
Y = np.loadtxt("Y.txt", delimiter=",", dtype="float64")

print('------ Training classifier with CV -------')
percentile = 60
imputer = SimpleImputer()
scaler = preprocessing.StandardScaler()
selector = SelectPercentile(mutual_info_regression, percentile=percentile)

#model = RandomForestClassifier(n_estimators = 1000,class_weight='balanced',min_samples_split=5,min_samples_leaf=2,max_features='sqrt',max_depth=20,bootstrap='False')
model = RandomForestClassifier(n_estimators = 1800,class_weight='balanced',
                               min_samples_split=10,min_samples_leaf=2,
                               max_features='auto',max_depth=50,bootstrap='False')

#model = SVC(class_weight='balanced')
model = Pipeline([
                ('imputer',imputer),
                ('standardizer', scaler),
                ('MI', selector),
                ('model', model)
                ])
# Fit the model
model.fit(X, Y)
# predict the testdata
print('predicting')
y_pred = model.predict(X_test)

print('writing to file')
with open('result.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['id', 'y'])
    for i in range(len(y_pred)):
        writer.writerow([i, y_pred[i]])