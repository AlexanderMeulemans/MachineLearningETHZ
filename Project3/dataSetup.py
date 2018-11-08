import numpy as np

X_len = 5118
X_test_len = 3412

X = []
X_test = []
for i in range(X_len-1):
    sample = np.genfromtxt("X_train.csv", delimiter=",", skip_header=1+i,skip_footer=X_len-i-2)
    X.append(sample[1:])
for i in range(X_test_len):
    sample = np.genfromtxt("X_test.csv", delimiter=",", skip_header=1 + i, skip_footer=X_test_len - i - 2)
    X_test.append(sample[1:])
y = np.genfromtxt("y_train.csv", delimiter=",", skip_header=1)
y = np.ravel(y[:,1:])
