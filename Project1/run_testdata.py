import numpy as np
from scipy import stats
from sklearn.feature_selection import VarianceThreshold, f_regression, mutual_info_regression, SelectPercentile
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
from data_setup import X,y,X_test
from helper_classes import OutlierExtractor
from helper_functions import cross_val_output, grid_search_output

# ===================== User variables =======================
variance_threshold = 0


# ======================= Preprocessing =======================
imputer = SimpleImputer()
X_imputed = imputer.fit_transform(X)
X_test_imputed = imputer.transform(X_test)
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# ======================= Feature selection ====================
variance_selector = VarianceThreshold(threshold=variance_threshold)
X_varianceSelected = variance_selector.fit_transform(X_scaled)
X_test_varianceSelected = variance_selector.transform(X_test_scaled)

