#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:58:01 2018

@author: martin
"""

import numpy as np
from sklearn.metrics import r2_score
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split,KFold,RepeatedKFold
from sklearn.impute import MissingIndicator
from scipy import stats
from sklearn.feature_selection import VarianceThreshold, f_regression, mutual_info_regression, SelectPercentile
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from data_setup import X,y, X_test
from helper_classes import OutlierExtractor
from helper_functions import cross_val_output
import matplotlib.pyplot as plt
import csv


"""
This is a pipeline that can be reused for other projects. The following steps are done: 
Feature selection
Fitting ML model
Testing ML model

The feature selection should optimize the training data in 3 ways: 
remove irrelevant features
remove outliers
take perturbations into account (e.g. missing values)
"""


def preprocess(X,y,X_test,variance_threshold,percentile):
    
    # Imputing the missing values
    imputer = SimpleImputer()
    X = imputer.fit_transform(X)
    X_test = imputer.fit_transform(X_test)
    
    # Standardize the data (because the features are very large, not optimal for ML)
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    scaler.fit(X_test)
    X_test = scaler.transform(X_test)
    
    #Select things that vary
    variance_selector = VarianceThreshold(threshold=variance_threshold).fit(X)
    X = variance_selector.transform(X)
    X_test = variance_selector.transform(X_test)
    
    
    #Select the best percentile of features
    Percentile = SelectPercentile(mutual_info_regression, percentile).fit(X,y)
    X = Percentile.transform(X)
    X_test = Percentile.transform(X_test)
    
    return X,X_test

def outlier_remove(X,y,outlier_threshold):
    
    #Remove outliers
    outlier_extractor = OutlierExtractor(neg_conf_val=outlier_threshold)
    X_outlier_removed,y_outlier_removed = outlier_extractor.transform(X,y)
    
    return X_outlier_removed, y_outlier_removed


# Pipeline user variables
name_outputfile = "outputfile2"
variance_threshold = 0
percentile = 2 # percentile of best features to be selected in the feature selection
outlier_threshold = -1.1 # threshold used to remove outliers
splits = 5
model = SVR()

#%%
X,X_test = preprocess(X,y,X_test,variance_threshold,percentile)
X,y = outlier_remove(X,y,outlier_threshold)
model.fit(X,y)

#%%
y_pred = model.predict(X_test)

#%%
with open('result.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['id','y'])
    for i in range(len(y_pred)):
        writer.writerow([i,y_pred[i]])





