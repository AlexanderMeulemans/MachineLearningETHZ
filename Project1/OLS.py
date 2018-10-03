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
import numpy as np
from scipy import stats
from sklearn.feature_selection import VarianceThreshold, f_regression, mutual_info_regression, SelectPercentile
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from data_setup import X,y
from helper_classes import OutlierExtractor
from helper_functions import cross_val_output
import matplotlib.pyplot as plt


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

# Pipeline user variables
name_outputfile = "outputfile2"
variance_threshold = 0
percentile = 2 # percentile of best features to be selected in the feature selection
outlier_threshold = -1.1 # threshold used to remove outliers


#%%
f = open(name_outputfile, 'w+')
f.write('========== General ==========\n')
f.write('Number of features: ')
f.write(str(X.shape[1]))
f.write('\n')
f.write('Number of training samples: ')
f.write(str(X.shape[0]))
f.write('\n')
f.close()

# -------------------------- PREPROCESSING -----------------------------
# Imputing the missing values
imputer_test = SimpleImputer()
X_imputed = imputer_test.fit_transform(X)
imputer = SimpleImputer()
# Standardize the data (because the features are very large, not optimal for ML)
scaler_test = preprocessing.StandardScaler().fit(X_imputed)
X_scaled = scaler_test.transform(X_imputed)
scaler = preprocessing.StandardScaler() # will be used later on in the cross validation pipeline

# --------------------- FEATURE SELECTION METHODS ----------------------
# Variance Threshold
variance_selector_test = VarianceThreshold(threshold=variance_threshold) # used for knowing how many features are selected
variance_selector = VarianceThreshold(threshold=variance_threshold) # used later on in cross validation pipeline
X_variance_selected = variance_selector_test.fit_transform(X_scaled)
X_var = variance_selector_test.variances_

f = open(name_outputfile, 'a')
f.write('========== Feature selection ==========\n')
f.write('Variance threshold ________________\n')
f.write('number of selected features: ')
f.write(str(X_variance_selected.shape[1]))
f.write('\n')
f.close()


# Mutual information
X_MI_selected = SelectPercentile(mutual_info_regression, percentile).fit_transform(X_variance_selected, y)
MI_selector = SelectPercentile(mutual_info_regression, percentile)  # will be used later on in the cross validation pipeline

f = open(name_outputfile, 'a')
f.write('Mutual information ________________ \n')
f.write('number of selected features: ')
f.write(str(X_MI_selected.shape[1]))
f.write('\n')
f.close()

# ---------------------- ML methods and pipelines --------------------------
#%% Ordinary Least Square
OLS = linear_model.LinearRegression()
OLS_pipeline = Pipeline([('imputer', imputer), ('standardizer', scaler),
                             ('variance', variance_selector), ('MI', MI_selector),
                             ('linreg', OLS)])
cross_val_output(X,y,OLS_pipeline,'OLS: ', name_outputfile,cv = 5)
#%% RIdge regression
Ridge = linear_model.RidgeCV()
Ridge_pipeline = Pipeline([('imputer', imputer), ('standardizer', scaler),
                             ('variance', variance_selector), ('MI', MI_selector),
                             ('ridge', Ridge)])
cross_val_output(X,y,Ridge_pipeline,'Ridge: ', name_outputfile,cv = 5)

#%% Elastic Net

Net = linear_model.ElasticNet()
Net_pipeline = Pipeline([('imputer', imputer), ('standardizer', scaler),
                             ('variance', variance_selector), ('MI', MI_selector),
                             ('Elastic Net', Net)])


cross_val_output(X,y,Net_pipeline,'Net: ', name_outputfile,cv = 5)

#%% BayesianRidge
Bay = linear_model.BayesianRidge()
Bay_pipeline = Pipeline([('imputer', imputer), ('standardizer', scaler),
                             ('variance', variance_selector), ('MI', MI_selector),
                             ('Bay', Bay)])


cross_val_output(X,y,Bay_pipeline,'BayesianRidge: ', name_outputfile,cv = 5)

#%% BayesianRidge
SVM = SVR()
SVM_pipeline = Pipeline([('imputer', imputer), ('standardizer', scaler),
                             ('variance', variance_selector), ('MI', MI_selector),
                             ('SVM', SVM)])


cross_val_output(X,y,SVM_pipeline,'SVM: ', name_outputfile,cv = 5)


#%%
#K = [0.1, 0.5, 1, 2, 5]
#rel = [0.5, 1, 2]
#for k in K:
#    for r in rel:
#        Net = linear_model.ElasticNet(alpha = k,l1_ratio=r)
#        Net_pipeline = Pipeline([('imputer', imputer), ('standardizer', scaler),
#                                     ('variance', variance_selector), ('MI', MI_selector),
#                                     ('Elastic Net', Net)])
#        f = open(name_outputfile, 'a')
#        f.write('alpha: %s\n l1_ratio: %s \n' %(k,r))
#        
#        cross_val_output(X,y,Net_pipeline,'Net: ', name_outputfile,cv = 5)
