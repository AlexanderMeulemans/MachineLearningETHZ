from dataSetup import dataSetup
from featureSelection import extractFeatures, extractFeatures2
from sklearn.feature_selection import mutual_info_regression, SelectPercentile
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer

import csv
import numpy as np

print('------ opening files -------')
X = np.loadtxt("X.txt", delimiter=",", dtype="float64")
X_test = np.loadtxt("X_test.txt", delimiter=",", dtype="float64")
Y = np.loadtxt("Y.txt", delimiter=",", dtype="float64")

print('------ Training classifier with CV -------')
percentile = 100
imputer = SimpleImputer()
scaler = preprocessing.StandardScaler()
selector = SelectPercentile(mutual_info_regression,percentile=percentile)
svc = SVC(kernel='rbf', class_weight='balanced')

pipeline = Pipeline([('imputer',imputer),('standardizer', scaler),
                     ('MI', selector),
                     ('svc', svc)])

Y_pred = cross_val_predict(pipeline, X, Y, cv=10)
score = f1_score(Y, Y_pred, average='micro')

print('average CV F1 score: ' + str(score))

print('------ Training classifier on total data -------')
pipeline.fit(X,Y)

print('------ Predicting test data -------')
Y_test_pred = pipeline.predict(X_test)
with open('result.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['id','y'])
    for i in range(len(Y_test_pred)):
        writer.writerow([i, Y_test_pred[i]])