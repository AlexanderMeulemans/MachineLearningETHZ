from sklearn.feature_selection import mutual_info_regression, SelectPercentile
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import f1_score, make_scorer, balanced_accuracy_score
from sklearn.impute import SimpleImputer
import sklearn.ensemble as skl
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import imblearn.ensemble as imb
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pandas as pd
from alex_featureselection import feature_extractor_eeg
import csv
from alex_classifier import AlexClassifier


print('------ opening files -------')
X = pd.read_csv('train_eeg1.csv', sep=',', index_col=0)
X = np.asarray(X)
X = feature_extractor_eeg(X)
# %%
Y = pd.read_csv('train_labels.csv', sep=',', index_col=0)
Y = np.asarray(Y)
Y = np.ravel(Y)

X_test = pd.read_csv('test_eeg1.csv', sep=',', index_col=0)
X_test = np.asarray(X_test)
X_test = feature_extractor_eeg(X_test)

print('------ Training classifier with CV -------')
percentile = 100
imputer = SimpleImputer()
scaler = preprocessing.StandardScaler()
selector = SelectPercentile(mutual_info_regression, percentile=percentile)

model = skl.RandomForestClassifier(class_weight='balanced')
pipeline_base = Pipeline([
    ('standardizer', scaler),
    #('MI', selector),
    ('model', model)
])

model2 = skl.RandomForestClassifier(class_weight='balanced')
pipeline_phase1 = Pipeline([
    ('standardizer', scaler),
    #('MI', selector),
    ('model', model)
])
