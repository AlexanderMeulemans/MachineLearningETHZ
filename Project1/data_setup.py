import numpy as np

y = np.genfromtxt("y_train.csv", delimiter=",", skip_header=1,filling_values=0)
X = np.genfromtxt("X_train.csv", delimiter=",", filling_values=0,skip_header=1)
y = np.ravel(y[:,1:])
X = X[:,1:]
