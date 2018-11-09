import numpy as np
from featureSelection import extractFeatures
from dataSetup import dataSetup

X_len = 5118
X_test_len = 3412


X,X_test,Y = dataSetup()

# X = []
# sample = np.genfromtxt("X_train.csv", delimiter=",", skip_header=6,skip_footer=X_len-7)
# X.append(sample[1:])
# X_features = extractFeatures(X,show=True)
