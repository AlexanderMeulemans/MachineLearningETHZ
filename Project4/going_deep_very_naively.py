import tensorflow as tf
import numpy as np
from keras.models import Sequential
from sklearn import preprocessing
from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
# from extract_naive_data import X_train, X_test, y_train, squeeze_y, X_test_raw
from extract_data_multiple_channels import X_train, X_test, y_train, squeeze_y, X_test_raw
import csv

# scale data
scaler = preprocessing.StandardScaler()
X_train_reshaped = np.reshape(X_train,(X_train.shape[0],X_train.shape[1]*X_train.shape[2]*X_train.shape[3]))
X_test_reshaped = np.reshape(X_test,(X_test.shape[0],X_test.shape[1]*X_test.shape[2]*X_test.shape[3]))
# X_val_reshaped = np.reshape(X_val,(X_val.shape[0],X_val.shape[1]*X_val.shape[2]))
X_train_reshaped = scaler.fit_transform(X_train_reshaped)
X_test_reshaped = scaler.transform(X_test_reshaped)
# X_val_reshaped = scaler.transform(X_val_reshaped)
X_train = np.reshape(X_train_reshaped,X_train.shape)
X_test = np.reshape(X_test_reshaped,X_test.shape)
# X_val = np.reshape(X_val_reshaped, X_val.shape)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

conv_net = Sequential()

# convolution layer 1
conv_net.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100,100,2)))
conv_net.add(MaxPooling2D(pool_size=(3,3)))
conv_net.add(Dropout(0.5))

print(conv_net.output.shape)

# convolution layer 2
conv_net.add(Conv2D(64, (3, 3), activation='relu'))
conv_net.add(MaxPooling2D(pool_size=(3,3)))
conv_net.add(Dropout(0.5))

print(conv_net.output.shape)

# fully connected
conv_net.add(Flatten())
conv_net.add(Dense(128, activation='relu'))
conv_net.add(Dropout(0.5))
conv_net.add(Dense(1, activation='sigmoid'))

conv_net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

conv_net.fit(X_train, y_train,batch_size=32, nb_epoch=7, verbose=1)

# score = conv_net.evaluate(X_val, y_val, verbose=0)
# print("%s: %.2f%%" % (conv_net.metrics_names[1], score[1]*100))

y_pred = conv_net.predict(X_test, verbose=1)

y_pred = squeeze_y(y_pred, X_test_raw)

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