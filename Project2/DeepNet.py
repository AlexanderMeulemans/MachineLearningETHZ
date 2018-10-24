from data_setup import X,y
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from helper_functions import class2sample_weights
import numpy as np

print(tf.__version__)

# Preprocessing data
X = StandardScaler().fit_transform(X)

# Cross validation loop
seed = 7
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
class_weights = {0:6., 1:1., 2:6.}

for train, val in kfold.split(X, y):
    # Design network
    model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(3, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Fit the model
    model.fit(X[train], y[train], epochs=150, batch_size=10, verbose=0, class_weight=class_weights)
    # evaluate the model
    sample_weights = class2sample_weights(y[val], class_weights)
    scores = model.evaluate(X[val], y[val], verbose=0, sample_weight = sample_weights)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(np.array(cvscores)), np.std(np.array(cvscores))))

