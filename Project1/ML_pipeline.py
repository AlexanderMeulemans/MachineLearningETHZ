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
name_outputfile = "outputfile1"
variance_threshold = 0
percentile = 40  # percentile of best features to be selected in the feature selection
outlier_threshold = -1.2 # threshold used to remove outliers
plots = True


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

if plots:
    plt.figure()
    ind = np.arange(len(X_var))
    plt.bar(ind, X_var)
    plt.ylabel('Variance')
    plt.title('Variance of the features')
    plt.show()

# Mutual information
X_MI_selected = SelectPercentile(mutual_info_regression, percentile).fit_transform(X_variance_selected, y)
MI_selector = SelectPercentile(mutual_info_regression, percentile)  # will be used later on in the cross validation pipeline

f = open(name_outputfile, 'a')
f.write('Mutual information ________________ \n')
f.write('number of selected features: ')
f.write(str(X_MI_selected.shape[1]))
f.write('\n')
f.close()

if plots:
    MI = mutual_info_regression(X_variance_selected,y)
    ind = np.arange(len(MI))
    plt.figure()
    plt.bar(ind, MI)
    plt.ylabel('MI')
    plt.title('Mutual information of each')
    plt.show()

# ------------------- Outlier detection ----------------------

# fit the model for outlier detection (default)
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# use fit_predict to compute the predicted labels of the training samples
# (when LOF is used for outlier detection, the estimator has no predict,
# decision_function and score_samples methods).
pred_outliers = clf.fit_predict(X_MI_selected)  # -1 if outlier, 1 if inlier
outlier_scores = clf.negative_outlier_factor_

outlier_extractor = OutlierExtractor(neg_conf_val=outlier_threshold)
X_outlier_removed,y_outlier_removed = outlier_extractor.transform(X_MI_selected,y)

if plots:
    ind = np.arange(len(outlier_scores))
    plt.figure()
    plt.bar(ind, outlier_scores)
    plt.ylabel('scores')
    plt.title('outlier scores (outlier has value close to zero)')
    plt.show()

f = open(name_outputfile, 'a')
f.write('Outlier removal ________________ \n')
f.write('Number of selected samples: ')
f.write(str(X_outlier_removed.shape[0]))
f.write('\n')
f.write('Threshold used: ')
f.write(str(outlier_threshold))
f.write('\n')
f.close()

# ---------------------- ML methods and pipelines --------------------------

# Defining models and pipelines
svr_rbf = SVR(kernel='rbf')
svr_lin = SVR(kernel='linear')
svr_poly = SVR(kernel='poly')



svr_rbf_pipeline = Pipeline([('imputer', imputer), ('standardizer', scaler),
                             ('variance', variance_selector), ('MI', MI_selector),
                             ('svr', svr_rbf)])
svr_lin_pipeline = Pipeline([('imputer', imputer), ('standardizer', scaler),
                             ('variance', variance_selector), ('MI', MI_selector),
                             ('svr', svr_lin)])
svr_poly_pipeline = Pipeline([('imputer', imputer), ('standardizer', scaler),
                              ('variance', variance_selector), ('MI', MI_selector),
                             ('svr', svr_poly)])

# Run models in a cross validation
cross_val_output(X,y,svr_rbf_pipeline,'SVR RBF', name_outputfile,cv = 5)
cross_val_output(X,y,svr_lin_pipeline,'SVR LIN', name_outputfile,cv = 5)
cross_val_output(X,y,svr_poly_pipeline,'SVR POLY', name_outputfile,cv = 5)

# Grid search
param_grid_svr_rbf = {
    'svr__C': stats.expon(scale=100),
    'svr__gamma': stats.expon(scale=.1)}

param_grid_svr_lin = {
    'svr__C': stats.expon(scale=100)}

param_grid_svr_poly = {
    'svr__C': stats.expon(scale=100),
    'svr__gamma': [2, 3, 4]}

search_svr_rbf = RandomizedSearchCV(svr_rbf_pipeline, param_grid_svr_rbf, cv=5)
search_svr_rbf.fit(X, y)
print("Best parameter (CV score=%0.3f):" % search_svr_rbf.best_score_)
print(search_svr_rbf.best_params_)













