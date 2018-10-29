from data_setup import X,y, X_test
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, SelectPercentile
from sklearn.model_selection import StratifiedKFold
from helper_functions import clip_resultfile
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

print(tf.__version__)

MI_percentile = 75.7

# Preprocessing data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

scaler = StandardScaler()
MI_selector = SelectPercentile(mutual_info_regression,percentile = MI_percentile)
svc = SVC(kernel='rbf', class_weight='balanced')

model = Pipeline([('standardizer', scaler),
                     ('MI', MI_selector),
                     ('svc', svc)])

# Fit the model
model.fit(X, y)
# predict the testdata
y_pred = model.predict(X_test)


with open('result.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['id', 'y'])
    for i in range(len(y_pred)):
        writer.writerow([i, y_pred[i]])