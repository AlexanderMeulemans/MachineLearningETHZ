from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plot = False


# ------------ USEFUL FUNCTIONS -----------------
def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64 , activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model


train_data = pd.read_csv("X_train.csv")
del train_data["id"]
train_data = train_data.fillna(train_data.mean())


train_labels = pd.read_csv("Y_train.csv")
train_labels = train_labels["y"]




# print(train_data)
# mean = train_data.mean(axis=0)
# std = train_data.std(axis=0)
# print(mean[:10], std[:10])
# train_data = (train_data - mean) / std

train_data = train_data.values
train_labels = train_labels.values

print(train_data.shape())


boston_housing = keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

print(train_data)



model = build_model()
model.summary()

EPOCHS = 5000
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])
plot_history(history)


[loss, mae] = model.evaluate(train_data, train_labels, verbose=0)
print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

# test_predictions = model.predict(test_data).flatten()

# plt.figure(2)
# plt.scatter(test_labels, test_predictions)
# plt.xlabel('True Values [1000$]')
# plt.ylabel('Predictions [1000$]')
# plt.axis('equal')
# plt.xlim(plt.xlim())
# plt.ylim(plt.ylim())
# _ = plt.plot([-100, 100], [-100, 100])

# plt.figure(3)
# error = test_predictions - test_labels
# plt.hist(error, bins = 50)
# plt.xlabel("Prediction Error [1000$]")
# _ = plt.ylabel("Count")
if plot: plt.show()
