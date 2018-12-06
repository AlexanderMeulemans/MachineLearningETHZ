from dataSetup import dataSetup
from featureSelection import extractFeatures, extractFeatures2
from sklearn.feature_selection import mutual_info_regression, SelectPercentile
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer

import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Dropout, Dense
from tensorflow.python.keras.utils import np_utils

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
X = scaler.fit_transform(imputer.fit_transform(X))


def create_model(optimizer='adagrad',
                 kernel_initializer='glorot_uniform',
                 dropout=0.2):
    model = Sequential()
    model.add(Dense(1024, activation='relu',kernel_initializer=kernel_initializer))
    model.add(Dropout(dropout))
    model.add(Dense(512, activation='relu',kernel_initializer=kernel_initializer))
    model.add(Dropout(dropout))
    model.add(Dense(4, activation='softmax', kernel_initializer=kernel_initializer))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

earlyStopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
class_weights = class_weight.compute_class_weight('balanced', np.unique(Y), Y)

# kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=np.random.seed(7))
# Y_pred = np.zeros(len(Y))
# for train, test in kfold.split(X, Y):
#     model = create_model()

model = create_model()
model.fit(X, dummy_y, epochs=30, batch_size=15, verbose=1,
          class_weight=class_weights, callbacks=[earlyStopping])
# Y_pred[test] = np.argmax(model.predict(X[test]), axis=1)

y_pred = model.predict(X_test)
# print("---- scoring ----")
# score = f1_score(Y, Y_pred, average='micro')
# print('average CV F1 score: ' + str(score))

print('writing to file')
with open('result.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['id', 'y'])
    for i in range(len(y_pred)):
        writer.writerow([i, np.argmax(y_pred[i])])

