import numpy as np
from sklearn.model_selection import train_test_split

X_train_raw = np.load('./data_alex/x_train.npy')
X_test_raw = np.load('./data_alex/x_test.npy')
y_train_raw = np.load('./data_alex/y_train.npy')
X_train_raw_BG = np.load('./data_alex/x_train.npy')
X_test_raw_BG = np.load('./data_alex/x_test.npy')

def compute_differences(X):
    X_difference = []
    for i, sample in enumerate(X):
        sample_difference = np.empty((sample.shape[0]-1,sample.shape[1],sample.shape[2]))
        for j in range(sample.shape[0]-1):
            sample_difference[j,:,:] = sample[j+1,:,:] - sample[j,:,:]
        X_difference.append(sample_difference)
    return X_difference

X_train2_raw = compute_differences(X_train_raw_BG)
X_test2_raw = compute_differences(X_test_raw_BG)


X_train = np.array([])
X_test = np.array([])
y_train = np.array([])

for i, sample in enumerate(X_train_raw):
    if i == 0:
        X_train = sample
    else:
        X_train = np.append(X_train, sample, 0)
    y_train = np.append(y_train, y_train_raw[i]*np.ones(sample.shape[0]))
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))

for i, sample in enumerate(X_test_raw):
    if i == 0:
        X_test = sample
    else:
        X_test = np.append(X_test, sample, 0)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

for i, sample in enumerate(X_train2_raw):
    sample = np.append(np.reshape(sample[0, :, :], (1, sample.shape[1], sample.shape[2])), sample, 0)# duplicate the first timepoint to have the same amount of timepoints in the optical flow
    if i == 0:
        X_train2 = sample
    else:
        X_train2 = np.append(X_train2, sample, 0)
X_train2 = np.reshape(X_train2, (X_train2.shape[0], X_train2.shape[1], X_train2.shape[2], 1))

for i, sample in enumerate(X_test2_raw):
    sample = np.append(np.reshape(sample[0, :, :],(1,sample.shape[1],sample.shape[2])), sample, 0)
    if i == 0:
        X_test2 = sample
    else:
        X_test2 = np.append(X_test2, sample, 0)

X_test2 = np.reshape(X_test2, (X_test2.shape[0], X_test2.shape[1], X_test2.shape[2], 1))


X_train = np.append(X_train,X_train2,3)
X_test = np.append(X_test, X_test2,3)







# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.15)


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



