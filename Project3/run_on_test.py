#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 16:25:52 2018

@author: martin
"""
from sklearn.feature_selection import mutual_info_regression, SelectPercentile
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from imblearn.pipeline import Pipeline
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier,AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
import numpy as np
from sklearn.preprocessing import LabelEncoder
import csv
from imblearn.over_sampling import SMOTE


print('------ opening files -------')
X = np.loadtxt("X4.txt", delimiter=",", dtype="float64")
X_test = np.loadtxt("X_test4.txt", delimiter=",", dtype="float64")
Y = np.loadtxt("Y4.txt", delimiter=",", dtype="float64")


print('------ Training classifier with CV -------')
percentile = 60

print('------ Training classifier -------')
percentile = 70
imputer = SimpleImputer()
scaler = preprocessing.StandardScaler()
selector = SelectPercentile(mutual_info_regression, percentile=percentile)
over_sample = SMOTE()
{'model__n_estimators': 780, 'model__min_samples_split': 2, 'model__min_samples_leaf': 4, 'model__max_features': 'sqrt', 'model__max_depth': None, 'model__bootstrap': False, 'MI__percentile': 100}
#model = RandomForestClassifier(n_estimators = 1000,class_weight='balanced',min_samples_split=5,min_samples_leaf=2,max_features='sqrt',max_depth=20,bootstrap='False')

model = RandomForestClassifier(n_estimators = 1800,class_weight='balanced',
                               min_samples_split=10,min_samples_leaf=2,
                               max_features='auto',max_depth=50,bootstrap='False')

#model = RandomForestClassifier(n_estimators = 200,class_weight='balanced',min_samples_split=2,min_samples_leaf=2,max_features='auto',max_depth=50,bootstrap='False')
model1 = RandomForestClassifier(n_estimators = 780,class_weight='balanced',min_samples_split=2,min_samples_leaf=4,max_features='sqrt',max_depth=None,bootstrap='False')
model2 = GradientBoostingClassifier()
model3 = GaussianProcessClassifier()
model = VotingClassifier(estimators=[('rfc',model1),('gb',model2),('gpc',model3)])
#model = AdaBoostClassifier()

#model = SVC(class_weight='balanced')
model = Pipeline([
                ('imputer',imputer),
                ('standardizer', scaler),
                ('over_sample',over_sample),
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