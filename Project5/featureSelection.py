# import sys
# sys.path.append(r"c:\users\alexander\appdata\local\programs\python\python36\lib\site-packages")
# sys.path.append(r"C:\Users\Alexander\Anaconda3\pkgs\pywavelets-1.0.1-py36h8c2d366_0\Lib\site-packages")
import numpy as np
# import pywt


def extractFeatures9(X):
    Fs = 128
    


def extractFeatures(X, show=False):
    Fs = 300 #Hz
    X_all_features = np.empty((len(X), 13))
    for i in range(0, len(X)):
        X_sample = X[i]
        X_summary = ecg.ecg(signal=X_sample, sampling_rate=Fs, show=show)
        # Start extracting features
        X_heart = ecg.extract_heartbeats(signal=X_sample,
                        rpeaks=X_summary['rpeaks'], sampling_rate=Fs)

        X_features = [np.mean(X_summary['heart_rate']), np.var(X_summary['heart_rate'])]
        rpeaks_indices = X_summary['rpeaks']
        rpeaks = X_sample[rpeaks_indices]

        qpeaks, speaks = [], []
        for rpeak, heartbeat in zip(X_heart['rpeaks'], X_heart["templates"]):
            qpeaks += np.min(heartbeat[:rpeak])
            # speaks += np.min(heartbeat[rpeak:], default=0)

        for peaks in (rpeaks, qpeaks):
            X_features.append(np.mean(peaks))
            X_features.append(np.var(peaks))

        # Wavelet transform
        coefficients = pywt.wavedec(X_sample,'db4',level=6)
        for coeff in coefficients:
            X_features += [np.mean(np.power(coeff, 2))]
        X_all_features[i,:] = np.array(X_features)


    return X_all_features

def extractFeatures2(X, show=False):
    Fs = 300
    X_all_features = np.empty((len(X), 362))
    for i in range(0, len(X)):
        X_sample = X[i]
        X_summary = ecg.ecg(signal=X_sample, sampling_rate=Fs, show=show)
        # Start extracting features
        X_features = [np.mean(X_summary['heart_rate']), np.var(X_summary['heart_rate'])]
        X_features = np.array(X_features)
        template_mean = np.mean(X_summary['templates'], 0)
        template_var = np.var(X_summary['templates'], 0)
        X_features = np.concatenate((X_features, template_mean, template_var))
        X_all_features[i, :] = X_features

    return X_all_features

def extractFeatures3(X, show = False):
    Fs = 300
    X_all_features = np.empty((len(X), 368))
    for i in range(0, len(X)):
        X_sample = X[i]
        X_summary = ecg.ecg(signal=X_sample, sampling_rate=Fs, show=show)
        # Start extracting features
        X_features = [np.mean(X_summary['heart_rate']),
                      np.var(X_summary['heart_rate'])]
        X_features = np.array(X_features)
        template_mean = np.mean(X_summary['templates'], 0)
        template_var = np.var(X_summary['templates'], 0)
        # Wavelet transform
        coefficients = pywt.wavedec(X_sample, 'db4', level=5)
        X_wavelets = []
        for coeff in coefficients:
            X_wavelets += [np.mean(np.power(coeff, 2))]
        X_wavelets = np.array(X_wavelets)

        X_features = np.concatenate((X_features, X_wavelets, template_mean,
        template_var))
        X_all_features[i, :] = X_features


    return X_all_features

def extractFeatures4(X, show = False):
    Fs = 300
    X_all_features = np.empty((len(X), 371))
    for i in range(0, len(X)):
        X_sample = X[i]
        X_summary = ecg.ecg(signal=X_sample, sampling_rate=Fs, show=show)
        # Start extracting features
        X_features = [np.mean(X_summary['heart_rate']),
                      np.var(X_summary['heart_rate'])]
        rpeaks_indices = X_summary['rpeaks']
        rpeaks = X_sample[rpeaks_indices]
        X_features.append(np.mean(rpeaks))
        X_features.append(np.var(rpeaks))
        X_features = np.array(X_features)
        template_mean = np.mean(X_summary['templates'], 0)
        template_var = np.var(X_summary['templates'], 0)
        # Wavelet transform
        coefficients = pywt.wavedec(X_sample, 'db4', level=6)
        X_wavelets = []
        for coeff in coefficients:
            X_wavelets += [np.mean(np.power(coeff, 2))]
        X_wavelets = np.array(X_wavelets)

        X_features = np.concatenate((X_features, X_wavelets, template_mean,
        template_var))
        X_all_features[i, :] = X_features


    return X_all_features
