import numpy as np
import pandas as pd
from matplotlib import pyplot as pt
from sklearn.preprocessing import normalize
import math

Fs = 128

print('------ opening files -------')
X_eeg1 = pd.read_csv('train_eeg1.csv',sep=',',index_col=0)
X_eeg1 = np.asarray(X_eeg1)
X_eeg1 = normalize(X_eeg1)
X_eeg1 = np.fft.fft(X_eeg1)
X_eeg1 = np.abs(X_eeg1)

X_eeg2 = pd.read_csv('train_eeg2.csv',sep=',',index_col=0)
X_eeg2 = normalize(X_eeg2)
X_eeg2 = np.fft.fft(X_eeg2)
X_eeg2 = np.abs(X_eeg2)

X_emg = pd.read_csv('train_emg.csv',sep=',',index_col=0)
X_emg = np.asarray(X_emg)
X_emg = np.fft.fft(X_emg)
X_emg = np.abs(X_emg)


#%%
Y = pd.read_csv('train_labels.csv',sep=',',index_col=0)
Y = np.asarray(Y)
Y = np.ravel(Y)

X_test = pd.read_csv('test_eeg1.csv',sep=',',index_col=0)
X_test = np.asarray(X_test)
X_test = np.fft.fft(X_test)
X_test = np.abs(X_test)




def plot_overview(X,Y):
    X = 20*np.log(X)
    Fs = 128.
    f = np.arange(X.shape[1])*float(Fs/X.shape[1])
    X_s1 = X[0:21600]
    X_s2 = X[21600:43200]
    X_s3 = X[43200:]
    idx1 = Y==1
    idx2 = Y==2
    idx3 = Y==3

    pt.figure()
    pt.subplot(331)
    pt.plot(f, np.mean(X_s1[idx1[0:21600],:],0))
    pt.plot(f, np.mean(X_s1[idx1[0:21600],:],0)+np.std(X_s1[idx1[0:21600],:],0), 'r-')
    pt.plot(f, np.mean(X_s1[idx1[0:21600], :], 0) - np.std(X_s1[idx1[0:21600], :], 0),'r-')
    pt.title('Subject1, class 1')
    pt.subplot(332)
    pt.plot(f, np.mean(X_s2[idx1[21600:43200], :], 0))
    pt.plot(f, np.mean(X_s2[idx1[21600:43200], :], 0) + np.std(X_s2[idx1[21600:43200], :], 0), 'r-')
    pt.plot(f, np.mean(X_s2[idx1[21600:43200], :], 0) - np.std(X_s2[idx1[21600:43200], :], 0), 'r-')
    pt.title('Subject2, class 1')
    pt.subplot(333)
    pt.plot(f, np.mean(X_s3[idx1[43200:], :], 0))
    pt.plot(f, np.mean(X_s3[idx1[43200:], :], 0) + np.std(X_s3[idx1[43200:], :], 0), 'r-')
    pt.plot(f, np.mean(X_s3[idx1[43200:], :], 0) - np.std(X_s3[idx1[43200:], :], 0), 'r-')
    pt.title('Subject3, class 1')
    pt.subplot(334)
    pt.plot(f, np.mean(X_s1[idx2[0:21600], :], 0))
    pt.plot(f, np.mean(X_s1[idx2[0:21600], :], 0) + np.std(X_s1[idx2[0:21600], :], 0), 'r-')
    pt.plot(f, np.mean(X_s1[idx2[0:21600], :], 0) - np.std(X_s1[idx2[0:21600], :], 0), 'r-')
    pt.title('Subject1, class 2')
    pt.subplot(335)
    pt.plot(f, np.mean(X_s2[idx2[21600:43200], :], 0))
    pt.plot(f, np.mean(X_s2[idx2[21600:43200], :], 0) + np.std(X_s2[idx2[21600:43200], :], 0), 'r-')
    pt.plot(f, np.mean(X_s2[idx2[21600:43200], :], 0) - np.std(X_s2[idx2[21600:43200], :], 0), 'r-')
    pt.title('Subject2, class 2')
    pt.subplot(336)
    pt.plot(f, np.mean(X_s3[idx2[43200:], :], 0))
    pt.plot(f, np.mean(X_s3[idx2[43200:], :], 0) + np.std(X_s3[idx2[43200:], :], 0), 'r-')
    pt.plot(f, np.mean(X_s3[idx2[43200:], :], 0) - np.std(X_s3[idx2[43200:], :], 0), 'r-')
    pt.title('Subject3, class 2')
    pt.subplot(337)
    pt.plot(f, np.mean(X_s1[idx3[0:21600], :], 0))
    pt.plot(f, np.mean(X_s1[idx3[0:21600], :], 0) + np.std(X_s1[idx3[0:21600], :], 0), 'r-')
    pt.plot(f, np.mean(X_s1[idx3[0:21600], :], 0) - np.std(X_s1[idx3[0:21600], :], 0), 'r-')
    pt.title('Subject1, class 3')
    pt.subplot(338)
    pt.plot(f, np.mean(X_s2[idx3[21600:43200], :], 0))
    pt.plot(f, np.mean(X_s2[idx3[21600:43200], :], 0) + np.std(X_s2[idx3[21600:43200], :], 0), 'r-')
    pt.plot(f, np.mean(X_s2[idx3[21600:43200], :], 0) - np.std(X_s2[idx3[21600:43200], :], 0), 'r-')
    pt.title('Subject2, class 3')
    pt.subplot(339)
    pt.plot(f, np.mean(X_s3[idx3[43200:], :], 0))
    pt.plot(f, np.mean(X_s3[idx3[43200:], :], 0) + np.std(X_s3[idx3[43200:], :], 0), 'r-')
    pt.plot(f, np.mean(X_s3[idx3[43200:], :], 0) - np.std(X_s3[idx3[43200:], :], 0), 'r-')
    pt.title('Subject3, class 3')
    pt.show()


plot_overview(X_eeg1,Y)
plot_overview(X_eeg2,Y)
plot_overview(X_emg,Y)