import tensorflow as tf
import keras
import numpy as np
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D, ConvLSTM2D, LSTM
from extract_naive_data import y_train_clipped
import csv
from sklearn.model_selection import train_test_split

X_train = np.loadtxt('X_train_CNN')
X_test = np.loadtxt('X_test_CNN')
y_train = y_train_clipped

variance_selector = VarianceThreshold()
X_train = variance_selector.fit_transform(X_train)
X_test = variance_selector.transform(X_test)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = np.reshape(X_train,(158,22,-1))
X_test = np.reshape(X_test, (69,22,-1))

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.13)

model = Sequential()
model.add(LSTM(50, activity_regularizer = keras.regularizers.l2(0.01),input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=3), keras.callbacks.ReduceLROnPlateau(patience=1)]

model.fit(X_train, y_train, epochs =20, callbacks=callbacks, verbose=1, batch_size=32, validation_data=(X_val, y_val))


y_pred = model.predict(X_test, verbose=1)
y_pred = np.reshape(y_pred, (69))

print('writing to file')
with open('result.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['id', 'y'])
    for i in range(len(y_pred)):
        writer.writerow([i, y_pred[i]])



def clip_predictions(y):
    y_clipped = np.empty(y.shape)
    for i in range(y.shape[0]):
        if y[i]>0.5:
            y_clipped[i] = 0.999
        else:
            y_clipped[i] = 0.001
    return y_clipped

y_pred_clipped = clip_predictions(y_pred)

with open('result2.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['id', 'y'])
    for i in range(len(y_pred_clipped)):
        writer.writerow([i, y_pred_clipped[i]])