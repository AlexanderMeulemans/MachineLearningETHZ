#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:17:59 2018

@author: martin
"""

import numpy as np

from scipy import stats
from sklearn.feature_selection import VarianceThreshold, f_regression, mutual_info_regression, SelectPercentile
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVR,SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
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
name_outputfile = "outputfile1"
variance_threshold = 0
percentile = 40  # percentile of best features to be selected in the feature selection
outlier_threshold = -1.2 # threshold used to remove outliers
plots = False


# Create output file (delete old outputfile if it exists with the same name)
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
imputer = SimpleImputer()
# Standardize the data (because the features are very large, not optimal for ML)
scaler = preprocessing.StandardScaler() # will be used later on in the cross validation pipeline

# --------------------- FEATURE SELECTION METHODS ----------------------
# Variance Threshold
variance_selector = VarianceThreshold(threshold=variance_threshold) # used later 

# Mutual information
MI_selector = SelectPercentile(mutual_info_regression)  # will be used later on in the cross validation pipeline

# ---------------------- ML methods and pipelines --------------------------

# Defining models and pipelines
SVC_rbf = SVC(kernel='rbf')





SVM_pipeline = Pipeline([('imputer', imputer), ('standardizer', scaler),
                             ('variance', variance_selector), ('MI', MI_selector),
                             ('svr', SVC_rbf)])

# Run models in a cross validation
cross_val_output(X,y,svr_rbf_pipeline,'SVC RBF', name_outputfile,cv = 5)

#%%
# Grid search
param_grid_svr_rbf = {
    'MI__percentile': stats.gamma(a=2, scale=10),
    'svr__C': stats.gamma(a=2,scale=50),
    'svr__gamma': stats.gamma(a=2,scale=1/X_MI_selected.shape[1])}

param_grid_svr_lin = {
    'svr__C': stats.expon(scale=50)}

param_grid_svr_poly = {
    'svr__C': stats.expon(scale=50),
    'svr__gamma': [2, 3, 4]}

param_grid_GP = {
    'GP__alpha': stats.gamma(a=2,scale=1e-10)
}


search_svr_rbf = grid_search_output(X,y,svr_rbf_pipeline,param_grid_svr_rbf, 'SVR RBF', name_outputfile, 5 , 'r2', 20)
print('SVR_RBF done')
search_GP_rbf = grid_search_output(X,y,GP_rbf_pipeline,param_grid_GP, 'GP RBF', name_outputfile, 5 , 'r2', 10)
print('GP_RBF done')
search_GP_matern = grid_search_output(X,y,GP_matern_pipeline,param_grid_GP, 'GP Matern', name_outputfile, 5 , 'r2', 10)
print('GP_Matern done')
search_GP_constant = grid_search_output(X,y,GP_constant_pipeline,param_grid_GP, 'GP Constant', name_outputfile, 5 , 'r2', 10)
print('GP_Constnant done')
