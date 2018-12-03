import tensorflow as tf
import numpy as np
from keras.models import Sequential
from sklearn import preprocessing
from keras.layers import Dense, Flatten, Activation, Dropout, Conv2D, MaxPooling2D
# from extract_naive_data import X_train, X_test, y_train, squeeze_y, X_test_raw
from extract_naive_data import X_train, X_test, y_train, squeeze_y, X_test_raw, X_test_clipped, X_train_clipped, y_train_clipped
import csv
import numpy as np
import keras

# scale data
scaler = preprocessing.StandardScaler()
X_train_reshaped = np.reshape(X_train,(X_train.shape[0],X_train.shape[1]*X_train.shape[2]*X_train.shape[3]))
X_test_reshaped = np.reshape(X_test,(X_test.shape[0],X_test.shape[1]*X_test.shape[2]*X_test.shape[3]))
X_train_reshaped = scaler.fit_transform(X_train_reshaped)
X_test_reshaped = scaler.transform(X_test_reshaped)
X_train = np.reshape(X_train_reshaped,X_train.shape)
X_test = np.reshape(X_test_reshaped,X_test.shape)

X_train_clipped_reshaped = np.reshape(X_train_clipped, (X_train_clipped.shape[0]*X_train_clipped.shape[1],-1))
X_train_clipped_reshaped = scaler.fit_transform(X_train_clipped_reshaped)
X_train_clipped = np.reshape(X_train_clipped_reshaped,X_train_clipped.shape)

X_test_clipped_reshaped = np.reshape(X_test_clipped, (X_test_clipped.shape[0]*X_test_clipped.shape[1],-1))
X_test_clipped_reshaped = scaler.transform(X_test_clipped_reshaped)
X_test_clipped = np.reshape(X_test_clipped_reshaped,X_test_clipped.shape)



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


# Train CNN
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


conv_net.fit(X_train, y_train,batch_size=32, nb_epoch=3, verbose=1)

feature_extractor = Sequential()
feature_extractor.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100,100,1), weights=conv_net.layers[0].get_weights()))
feature_extractor.add(MaxPooling2D(pool_size=(3,3)))
# feature_extractor.add(Dropout(0.5))
feature_extractor.add(Conv2D(64, (3, 3), activation='relu', weights=conv_net.layers[3].get_weights()))
feature_extractor.add(MaxPooling2D(pool_size=(3,3)))

X_train_clipped_reshaped = np.reshape(X_train_clipped, (X_train_clipped.shape[0]*X_train_clipped.shape[1], 100, 100,1))
X_train_CNN_reshaped = feature_extractor.predict(X_train_clipped_reshaped, verbose=1)
X_train_CNN = np.reshape(X_train_CNN_reshaped,(X_train_clipped.shape[0]*X_train_clipped.shape[1],-1))

X_test_clipped_reshaped = np.reshape(X_test_clipped, (X_test_clipped.shape[0]*X_test_clipped.shape[1], 100, 100,1))
X_test_CNN_reshaped = feature_extractor.predict(X_test_clipped_reshaped, verbose=1)
X_test_CNN = np.reshape(X_test_CNN_reshaped,(X_test_clipped.shape[0]*X_test_clipped.shape[1],-1))

print(X_train_CNN.shape)

np.savetxt('X_train_CNN', X_train_CNN)
np.savetxt('X_test_CNN.txt', X_test_CNN)


