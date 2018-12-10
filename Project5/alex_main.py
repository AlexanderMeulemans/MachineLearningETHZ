def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

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
import os
from alex_masterplan import AlexClassifier
import alex_pipeline_utils

should_preprocess = False
preprocess_dir = "./preprocessed/"

X, X_test = (preprocess_data(preprocess_dir) if
                should_preprocess else load_data(preprocess_dir))

Y = pd.read_csv('train_labels.csv', sep=',', index_col=0)
Y = np.ravel(np.asarray(Y))

print('\n********* Training AlexClassifier')
model_alex = AlexClassifier(depth=3)
model_alex.fit(X,Y)

y_prob = model_alex.predict(X)
score = balanced_accuracy_score(Y, y_prob)
print('average CV F1 score: ' + str(score))

y_pred = model_alex.predict(X_test)

print('\n********* Writing to file')
with open('result.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['Id', 'y'])
    for i in range(len(y_pred)):
        writer.writerow([i, y_pred[i]])
