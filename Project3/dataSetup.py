import numpy as np
import pandas as pd

data_dir = "./data/"

X = pd.read_csv(data_dir + "test_eeg1.csv", sep=",", header=0, index_col=0)



def dataSetup():
    # print("reading csv ...")
    X_pd = pd.read_csv("X_train.csv", sep=",", header=0, index_col=0)
    # print("converting to list ...")
    X_lst = X_pd.values.tolist()
    # print('converting to array ...')
    X = []
    for lst in X_lst:
        array = np.array(lst)
        X.append(array[np.logical_not(np.isnan(array))])
    X_test_pd = pd.read_csv("X_test.csv", sep=",", header=0, index_col=0)
    X_test_lst = X_test_pd.values.tolist()
    X_test = []
    for lst in X_test_lst:
        array = np.array(lst)
        X_test.append(array[np.logical_not(np.isnan(array))])
    y = np.genfromtxt("y_train.csv", delimiter=",", skip_header=1)
    y = np.ravel(y[:, 1:])

    return X, y, X_test
