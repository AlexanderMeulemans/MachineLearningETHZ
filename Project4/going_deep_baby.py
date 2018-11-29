import os
import tensorflow as tf
import numpy as np
from tf_utils import input_fn_from_dataset,input_fn_frame_from_dataset,save_tf_record,prob_positive_class_from_prediction
from get_data import get_videos_from_folder,get_target_from_csv
from utils import save_solution

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Dropout, Dense, Conv3D, Conv2D, MaxPooling2D, TimeDistributed, LSTM, Flatten
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import roc_auc_score

print('------ opening files -------')
dir_path = os.path.dirname(os.path.realpath(__file__))
train_folder = os.path.join(dir_path,"./train/")
test_folder = os.path.join(dir_path,"./test/")

train_target = os.path.join(dir_path,'./train_target.csv')
my_solution_file = os.path.join(dir_path,'./solution.csv')

y = get_target_from_csv(train_target)
X_train = get_videos_from_folder(train_folder)
x_test = get_videos_from_folder(test_folder)

def build_cnn():
    model = Sequential()
    model.add(Conv2D(128, 3, input_shape=(100, 100, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    return model

def build_lstm(cnn):
    model = Sequential()
    model.add(TimeDistributed(cnn))
    model.add(LSTM(5, input_shape=(50, 50)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


time_size = 22
img_width = 100
img_height = 100
img_channels = 1
number_of_epochs = 10

def model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(64, (3,3), activation='relu'),
                              input_shape=(time_size, img_width, img_height, img_channels)))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    model.add(TimeDistributed(Conv2D(128, (4,4), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(time_size, return_sequences=False, dropout=0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()

    return model

#
# def lrcn():
#     model = Sequential()
#
#     model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2),
#                                      activation='relu', padding='same'), input_shape=(None, 100, 100, 1)))
#     model.add(TimeDistributed(Conv2D(32, (3, 3),
#                                      kernel_initializer="he_normal", activation='relu')))
#     model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
#
#     model.add(TimeDistributed(Conv2D(64, (3, 3),
#                                      padding='same', activation='relu')))
#     model.add(TimeDistributed(Conv2D(64, (3, 3),
#                                      padding='same', activation='relu')))
#     model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
#
#     model.add(TimeDistributed(Conv2D(128, (3, 3),
#                                      padding='same', activation='relu')))
#     model.add(TimeDistributed(Conv2D(128, (3, 3),
#                                      padding='same', activation='relu')))
#     model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
#
#     model.add(TimeDistributed(Conv2D(256, (3, 3),
#                                      padding='same', activation='relu')))
#     model.add(TimeDistributed(Conv2D(256, (3, 3),
#                                      padding='same', activation='relu')))
#     model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
#
#     model.add(TimeDistributed(Conv2D(512, (3, 3),
#                                      padding='same', activation='relu')))
#     model.add(TimeDistributed(Conv2D(512, (3, 3),
#                                      padding='same', activation='relu')))
#     model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
#
#     model.add(TimeDistributed(Flatten()))
#
#     model.add(Dropout(0.5))
#     model.add(LSTM(256, return_sequences=False, dropout=0.5))
#     model.add(Dense(1, activation='sigmoid'))
#
#     return model


net = model()
for e in range(number_of_epochs):
    net.fit(X_train, y, initial_epoch=e, verbose=1, batch_size=15)

    y_pred = net.predict(X_train)
    roc_auc = roc_auc_score(y, y_pred)

    for i in range(10):
        print(y_pred[i], y[i])

    print("\n\n loss: " + str(roc_auc))

print(net.evaluate(X_train, y))





