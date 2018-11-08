import sys
sys.path.append(r"c:\users\alexander\appdata\local\programs\python\python36\lib\site-packages")
sys.path.append(r"C:\Users\Alexander\Anaconda3\pkgs\pywavelets-1.0.1-py36h8c2d366_0\Lib\site-packages")

import numpy as np
from biosppy.signals import ecg
from pywt import wavedec
def extractFeatures(X):
    # X_len = 5118
    # X_test_len = 3412

    # X = []
    # sample = np.genfromtxt("X_train.csv", delimiter=",", skip_header=1,skip_footer=X_len-2)
    # X.append(sample[1:])
    Fs = 300 #Hz
    X_all_features = np.empty((len(X),8))
    for i in range(len(X)):
        X_sample = X[i]
        X_summary = ecg.ecg(signal=X_sample,sampling_rate = Fs,show =False)
        X_features = []
        X_features.append(np.mean(X_summary['heart_rate']))
        X_features.append(np.var(X_summary['heart_rate']))

        # Wavelet transform
        coeff = wavedec(X_sample,'db4',level=5)
        cA5,cD5, cD4, cD3, cD2, cD1 = coeff
        X_features.append(np.sum(np.power(cA5,2)))
        X_features.append(np.sum(np.power(cD5,2)))
        X_features.append(np.sum(np.power(cD4,2)))
        X_features.append(np.sum(np.power(cD3,2)))
        X_features.append(np.sum(np.power(cD2,2)))
        X_features.append(np.sum(np.power(cD1,2)))
        X_all_features[i,:] = np.array(X_features)
    return X_all_features



