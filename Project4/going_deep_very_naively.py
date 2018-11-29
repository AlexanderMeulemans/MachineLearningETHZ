import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
from keras.utils import np_utils
from extract_naive_data import X_train, X_val, X_test, y_train, y_val, squeeze_y, X_test_raw
import csv

conv_net = Sequential()

# convolution layer 1
conv_net.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100,100,1)))
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

conv_net.fit(X_train, y_train,batch_size=32, nb_epoch=40, verbose=1)

score = conv_net.evaluate(X_val, y_val, verbose=0)
print("%s: %.2f%%" % (conv_net.metrics_names[1], score[1]*100))

y_pred = conv_net.predict(X_test, verbose=1)

y_pred = squeeze_y(y_pred, X_test_raw)

print('writing to file')
with open('result.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['id', 'y'])
    for i in range(len(y_pred)):
        writer.writerow([i, y_pred[i]])

