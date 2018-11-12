# import sys
# sys.path.append(r"c:\users\alexander\appdata\local\programs\python\python36\lib\site-packages")
# sys.path.append(r"C:\Users\Alexander\Anaconda3\pkgs\pandas-0.23.4-py36h830ac7b_0\Lib\site-packages")
# sys.path.append(r"C:\Users\Alexander\Anaconda3\pkgs\pytz-2018.5-py36_0\Lib\site-packages")
import numpy as np
import pandas as pd

print("reading csv ...")
X_pd = pd.read_csv("X_train.csv",sep=",",header=0,index_col=0)
print("converting to list ...")
X_lst = X_pd.values.tolist()
print('converting to array ...')

X = []
for lst in X_lst:
    array = np.array(lst)
    X.append(array[np.logical_not(np.isnan(array))])
X_test_pd = pd.read_csv("X_test.csv",sep=",",header=0,index_col=0)
X_test_lst = X_test_pd.values.tolist()
X_test = []
for lst in X_test_lst:
    array = np.array(lst)
    X_test.append(array[np.logical_not(np.isnan(array))])
y = np.genfromtxt("y_train.csv", delimiter=",", skip_header=1)
y = np.ravel(y[:,1:])