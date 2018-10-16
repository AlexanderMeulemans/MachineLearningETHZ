from data_setup import X,y, X_test
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from helper_functions import clip_resultfile
import numpy as np
import matplotlib.pyplot as plt
import csv

print(tf.__version__)

# Preprocessing data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# Cross validation loop
class_weights = {0:6., 1:1., 2:6.}

# Design network
model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(3, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit the model
model.fit(X, y, epochs=150, batch_size=10, verbose=0, class_weight=class_weights)
# predict the testdata
y_pred = model.predict(X_test)
y_pred = clip_resultfile(y_pred)

with open('result.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['id', 'y'])
    for i in range(len(y_pred)):
        writer.writerow([i, y_pred[i]])