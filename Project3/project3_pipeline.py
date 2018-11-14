from dataSetup import dataSetup
from featureSelection import extractFeatures, extractFeatures2
from sklearn.feature_selection import mutual_info_regression, SelectPercentile
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer

import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dropout, Dense
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier

print('------ opening files -------')
X = np.loadtxt("X.txt", delimiter=",", dtype="float64")
X_test = np.loadtxt("X_test.txt", delimiter=",", dtype="float64")
Y = np.loadtxt("Y.txt", delimiter=",", dtype="float64")

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)

print('------ Training classifier with CV -------')
percentile = 100
imputer = SimpleImputer()
scaler = preprocessing.StandardScaler()
selector = SelectPercentile(mutual_info_regression, percentile=percentile)


def create_model(optimizer='adagrad',
                 kernel_initializer='glorot_uniform',
                 dropout=0.2):
    model = Sequential()
    model.add(Dense(1024, activation='relu',kernel_initializer=kernel_initializer))
    model.add(Dropout(dropout))
    model.add(Dense(64, activation='relu',kernel_initializer=kernel_initializer))
    model.add(Dropout(dropout))
    model.add(Dense(4, activation='softmax', kernel_initializer=kernel_initializer))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

pipeline = Pipeline([
                    ('imputer',imputer),
                    ('standardizer', scaler),
                    ('MI', selector),
                    ('keras', KerasClassifier(build_fn=create_model,epochs=30, batch_size=15,verbose=1))
                     ])

print("---- predicting ----")
Y_pred = cross_val_predict(pipeline, X, dummy_y, cv=2)

print("---- scoring ----")
score = f1_score(Y, Y_pred, average='micro')

print('average CV F1 score: ' + str(score))

# print('------ Training classifier on total data -------')
# pipeline.fit(X,Y)s

# print('------ Predicting test data -------')
# Y_test_pred = pipeline.predict(X_test)
# with open('result.csv', mode='w') as csv_file:
#     writer = csv.writer(csv_file, delimiter=',')
#     writer.writerow(['id','y'])
#     for i in range(len(Y_test_pred)):
#         writer.writerow([i, Y_test_pred[i]])