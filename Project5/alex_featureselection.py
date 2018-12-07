import numpy as np
import pandas as pd
from matplotlib import pyplot as pt
from sklearn.preprocessing import normalize
import math

def feature_extractor_eeg(X):
    Fs = 128.0
    X_len = float(X.shape[1])
    idx_dc = math.ceil(0.39/Fs*X_len)
    idx_delta = math.ceil(3.13/Fs*X_len)
    idx_theta = math.ceil(8.46/Fs*X_len)
    idx_alpha = math.ceil(10.93/Fs*X_len)
    idx_spindle = math.ceil(15.63/Fs*X_len)
    idx_beta1 = math.ceil(21.88/Fs*X_len)
    idx_beta2 = math.ceil(37.50/Fs*X_len)
    idx_gamma = math.ceil(X_len/2-1)

    X = np.fft.fft(X)
    X = np.abs(X)

    E_total = np.sum(np.power(X,2),1)
    E_total = np.reshape(E_total,(E_total.shape[0],1))
    X = normalize(X,axis=1)

    E_dc = np.sum(np.power(X[:,0:idx_dc],2),1)
    E_dc = np.reshape(E_dc, (E_dc.shape[0], 1))
    E_delta = np.sum(np.power(X[:,idx_dc:idx_delta],2),1)
    E_delta = np.reshape(E_delta, (E_total.shape[0], 1))
    E_theta = np.sum(np.power(X[:,idx_delta:idx_theta],2),1)
    E_theta = np.reshape(E_theta, (E_total.shape[0], 1))
    E_alpha = np.sum(np.power(X[:, idx_theta:idx_alpha], 2), 1)
    E_alpha = np.reshape(E_alpha, (E_total.shape[0], 1))
    E_spindle = np.sum(np.power(X[:,idx_alpha:idx_spindle],2),1)
    E_spindle = np.reshape(E_spindle, (E_total.shape[0], 1))
    E_beta1 = np.sum(np.power(X[:,idx_spindle:idx_beta1],2),1)
    E_beta1 = np.reshape(E_beta1, (E_total.shape[0], 1))
    E_beta2 = np.sum(np.power(X[:,idx_beta1:idx_beta2],2),1)
    E_beta2 = np.reshape(E_beta2, (E_total.shape[0], 1))
    E_gamma = np.sum(np.power(X[:,idx_beta2:idx_gamma],2),1)
    E_gamma = np.reshape(E_gamma, (E_total.shape[0], 1))
    E_ratio1 = np.divide(E_alpha, E_delta + E_theta)
    E_ratio2 = np.divide(E_delta, E_alpha + E_theta)
    E_ratio3 = np.divide(E_theta, E_alpha + E_delta)
    features = np.concatenate((E_total, E_dc, E_delta, E_theta, E_alpha, E_spindle, E_beta1, E_beta2, E_gamma,
                               E_ratio1, E_ratio2, E_ratio3), axis=1)
    return features

def feature_extractor_combined(X, emg):
    Fs = 128.0
    X_len = float(X.shape[1])
    idx_dc = math.ceil(0.39/Fs*X_len)
    idx_delta = math.ceil(3.13/Fs*X_len)
    idx_theta = math.ceil(8.46/Fs*X_len)
    idx_alpha = math.ceil(10.93/Fs*X_len)
    idx_spindle = math.ceil(15.63/Fs*X_len)
    idx_beta1 = math.ceil(21.88/Fs*X_len)
    idx_beta2 = math.ceil(37.50/Fs*X_len)
    idx_gamma = math.ceil(X_len/2-1)

    X = np.fft.fft(X)
    X = np.abs(X)

    E_total = np.sum(np.power(X,2),1)
    E_total = np.reshape(E_total,(E_total.shape[0],1))
    X = normalize(X,axis=1)

    E_dc = np.sum(np.power(X[:,0:idx_dc],2),1)
    E_dc = np.reshape(E_dc, (E_dc.shape[0], 1))
    E_delta = np.sum(np.power(X[:,idx_dc:idx_delta],2),1)
    E_delta = np.reshape(E_delta, (E_total.shape[0], 1))
    E_theta = np.sum(np.power(X[:,idx_delta:idx_theta],2),1)
    E_theta = np.reshape(E_theta, (E_total.shape[0], 1))
    E_alpha = np.sum(np.power(X[:, idx_theta:idx_alpha], 2), 1)
    E_alpha = np.reshape(E_alpha, (E_total.shape[0], 1))
    E_spindle = np.sum(np.power(X[:,idx_alpha:idx_spindle],2),1)
    E_spindle = np.reshape(E_spindle, (E_total.shape[0], 1))
    E_beta1 = np.sum(np.power(X[:,idx_spindle:idx_beta1],2),1)
    E_beta1 = np.reshape(E_beta1, (E_total.shape[0], 1))
    E_beta2 = np.sum(np.power(X[:,idx_beta1:idx_beta2],2),1)
    E_beta2 = np.reshape(E_beta2, (E_total.shape[0], 1))
    E_gamma = np.sum(np.power(X[:,idx_beta2:idx_gamma],2),1)
    E_gamma = np.reshape(E_gamma, (E_total.shape[0], 1))
    E_ratio1 = np.divide(E_alpha, E_delta + E_theta)
    E_ratio2 = np.divide(E_delta, E_alpha + E_theta)
    E_ratio3 = np.divide(E_theta, E_alpha + E_delta)
    features = np.concatenate((E_total, E_dc, E_delta, E_theta, E_alpha, E_spindle, E_beta1, E_beta2, E_gamma,
                               E_ratio1, E_ratio2, E_ratio3), axis=1)
    return features

