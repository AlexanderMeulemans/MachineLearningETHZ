import numpy as np
from sklearn.model_selection import train_test_split

X_train_raw = np.load('./data_alex/x_train.npy')
X_test_raw = np.load('./data_alex/x_test.npy')
y_train_raw = np.load('./data_alex/y_train.npy')

X_train = np.array([])
X_test = np.array([])
y_train = np.array([])

for i, sample in enumerate(X_train_raw):
    if i == 0:
        X_train = sample
    else:
        X_train = np.append(X_train, sample, 0)
    y_train = np.append(y_train, y_train_raw[i]*np.ones(sample.shape[0]))

for i, sample in enumerate(X_test_raw):
    if i == 0:
        X_test = sample
    else:
        X_test = np.append(X_test, sample, 0)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.15)

min_length = 100000
for sample in X_train_raw:
    min_length = min(min_length, sample.shape[0])
for sample in X_test_raw:
    min_length = min(min_length, sample.shape[0])

# clip all samples to minimum sample length
X_train_clipped = np.empty((len(X_train_raw),min_length,100,100))
X_test_clipped = np.empty((len(X_test_raw),min_length,100,100))

for i,sample in enumerate(X_train_raw):
    X_train_clipped[i,:,:,:] = sample[0:min_length,:,:]

for i,sample in enumerate(X_test_raw):
    X_test_clipped[i,:,:,:] = sample[0:min_length,:,:]







def squeeze_y(y,X_raw):
    y_squeezed = np.empty(len(X_raw))
    index = 0
    for i, x in enumerate(X_raw):
        x_len = x.shape[0]
        y_squeezed[i] = np.mean(y[index:index+x_len])
        index += x_len
    return y_squeezed

# def standardize(X, axis = 0):
#     mean = np.mean(X,axis)
#     std = np.std(X,axis)
#     X_standardized = np.empty(X.shape)
#     for i in range(X.shape[0]):
#         X_standardized[i,:,:,:] =
#     X_standardized = (X-mean)/std
#     return X_standardized


