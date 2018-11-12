from dataSetup import dataSetup
from featureSelection import extractFeatures2
import numpy as np

print('------ Loading data ------')
X_raw, Y, X_test_raw = dataSetup()

print('------ Extracting features ------')
X = extractFeatures2(X_raw)
X_test = extractFeatures2(X_test_raw)

np.savetxt("X.txt", X, delimiter=",")
np.savetxt("X_test.txt", X_test, delimiter=",")
np.savetxt("Y.txt", Y, delimiter=",")

print("X dtype: " + str(X.dtype))
print("X_test dtype: " + str(X_test.dtype))
print("Y dtype: " + str(Y.dtype))
