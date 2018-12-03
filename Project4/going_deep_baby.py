import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dropout, Dense, Conv3D, Conv2D, MaxPooling2D, TimeDistributed, LSTM, Flatten
from tensorflow.python.keras.models import load_model
from sklearn.metrics import roc_auc_score

SHOULD_LOAD = True
number_of_epochs = 1
batch_size = 10

time_size = 22
img_width = img_height = 100

# Data is formatted as (#batches, #frames, height, width, #channels)
X = np.load("X.npy")
X_test = np.load("X_test.npy")

# formatted as (#batches, classifier)
y = np.load("Y.npy")

def model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(64, (3,3), activation='relu'),
                              input_shape=(time_size, img_width, img_height, 1)))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(1, 1))))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(time_size))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

benchmark_model_name = 'model_lstm.h5'
if SHOULD_LOAD:
    net = load_model(benchmark_model_name)
else:
    net = model()
    net.fit(X, y, verbose=1, batch_size=7, nb_epoch=number_of_epochs)
    print('--- saving ---')
    net.save(benchmark_model_name)
    print('--- done ---')


# for Hydrogen usage
print(net.summary())
for layer in net.layers:
    print("\n\n\n")
    g=layer.get_config()
    h=layer.output
    print (g)
    print (h)

import matplotlib.pyplot as plt
print('--- evaluating ---')
y_pred = net.predict_proba(X, verbose=1)
for i in range(10):
    print(y_pred[i], y[i])

y_pred = net.predict(X[2:3,:,:,:,:], verbose=1)
print(y_pred)
print(y[2])




# roc_auc = roc_auc_score(y, y_pred)
# for i in range(10):
#     print(y_pred[i], y[i])
# print("\n\n loss: " + str(roc_auc))

# print(net.evaluate(X_train, y))
