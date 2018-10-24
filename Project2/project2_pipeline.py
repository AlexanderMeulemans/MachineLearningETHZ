import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression, SelectPercentile
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from Project2.helper_functions import cross_val_output

pycharm = False
variance_threshold = 0
percentile = 40
outlier_threshold = -1.2

#%%

print("OPENING FILES")
root = "Project2/" if pycharm else ""
Y = np.genfromtxt(root + "y_train.csv", delimiter=",", skip_header=1)
X = np.genfromtxt(root + "X_train.csv", delimiter=",", skip_header=1)

Y = np.ravel(Y[:, 1:])
X = np.array(X[:, 1:])



#%%

print('========== STARTED PREPROCESSING ==========\n')

scaler_test = preprocessing.StandardScaler().fit(X)
X_scaled = scaler_test.transform(X)
scaler = preprocessing.StandardScaler()


X_MI_selected = SelectPercentile(mutual_info_regression, percentile).fit_transform(X, Y)
MI_selector = SelectPercentile(mutual_info_regression)

print('Number of features: ' + str(X.shape[1]))
print('Number of training samples: ' + str(X.shape[0]))
print('Number of selected features through I(.): ' + str(X_MI_selected.shape[1]))


#%%
svc = SVC(kernel='rbf',class_weight='balanced')
svc_pipe = Pipeline([('standardizer', scaler),
                     ('MI', MI_selector),
                     ('svc', svc)])

cross_val_output(X, Y, svc_pipe, 'SVC', cv=5)
#%%
# Grid search
# param_grid_svr_rbf = {
#     'MI__percentile': stats.gamma(a=2, scale=10),
#     'svr__C': stats.gamma(a=2,scale=50),
#     'svr__gamma': stats.gamma(a=2,scale=1/X_MI_selected.shape[1])
# }
#
# param_grid_svr_lin = {
#     'svr__C': stats.expon(scale=50)
# }
#
# param_grid_svr_poly = {
#     'svr__C': stats.expon(scale=50),
#     'svr__gamma': [2, 3, 4]
# }
#
# param_grid_GP = {
#     'GP__alpha': stats.gamma(a=2,scale=1e-10)
# }

# search_svr_rbf = grid_search_output(X,y,svr_rbf_pipeline,param_grid_svr_rbf, 'SVR RBF', name_outputfile, 5 , 'r2', 20)
# print('SVR_RBF done')
# search_GP_rbf = grid_search_output(X,y,GP_rbf_pipeline,param_grid_GP, 'GP RBF', name_outputfile, 5 , 'r2', 10)
# print('GP_RBF done')
# search_GP_matern = grid_search_output(X,y,GP_matern_pipeline,param_grid_GP, 'GP Matern', name_outputfile, 5 , 'r2', 10)
# print('GP_Matern done')
# search_GP_constant = grid_search_output(X,y,GP_constant_pipeline,param_grid_GP, 'GP Constant', name_outputfile, 5 , 'r2', 10)
# print('GP_Constnant done')
