import numpy as np
import csv

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dropout, Dense, Conv3D, Conv2D, MaxPooling2D, TimeDistributed, LSTM, Flatten
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import RMSprop, Adam
from tensorflow.python.keras.callbacks import EarlyStopping

from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle


SHOULD_LOAD = False
number_of_epochs = 20
number_of_batches = 7

time_size = 20
img_width = img_height = 100

# Data is formatted as (#batches, #frames, height, width, #channels)
X = np.load("X.npy")
X_test = np.load("X_test.npy")
y = np.load("Y.npy")
vote_map = np.load("vote_map.npy")

X, y = shuffle(X, y, random_state=0)

def model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (3,3), activation='relu'),
                              input_shape=(None, img_width, img_height, 1)))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(32, (3,3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Flatten()))
    model.add(Dropout(0.5))

    model.add(LSTM(time_size, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
    model.summary()
    return model

benchmark_model_name = 'model_lstm.h5'
if SHOULD_LOAD:
    net = load_model(benchmark_model_name)
else:
    net = model()

    earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
    net.fit(X, y, batch_size=number_of_batches, epochs=number_of_epochs, verbose=1,
              callbacks=[earlyStopping], validation_split=0.33, shuffle=True)

    print('--- saving ---')
    net.save(benchmark_model_name)
    print('--- done ---')



y_pred = net.predict_proba(X, verbose=1)
for i in range(10):
    print(y_pred[i], int(np.around(y_pred[i])[0]), y[i])

roc_auc = roc_auc_score(y, y_pred)
print("\n\n auc score: " + str(roc_auc))


y_pred, index = [], 0
predictions = net.predict_proba(X_test, verbose=1)
for i in range(vote_map.shape[0]):
    N, votes = vote_map[i], []
    for j in range(N):
        votes += [predictions[index][0]]
        index += 1
    y_pred += [ sum(votes) / N ]

print(np.around(y_pred))



print('writing to file')
with open('result.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['id', 'y'])
    for i in range(len(y_pred)):
        writer.writerow([i, y_pred[i]])
print("done")
