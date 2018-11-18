from dataSetup import dataSetup
from featureSelection import extractFeatures4
import numpy as np



print('------ Loading data ------')
X_raw, Y, X_test_raw = dataSetup()

print('------ Extracting features ------')
X = extractFeatures4(X_raw)
X_test = extractFeatures4(X_test_raw)

np.savetxt("X4.txt", X, delimiter=",")
np.savetxt("X_test4.txt", X_test, delimiter=",")
np.savetxt("Y4.txt", Y, delimiter=",")

print("X dtype: " + str(X.dtype))
print("X_test dtype: " + str(X_test.dtype))
print("Y dtype: " + str(Y.dtype))
