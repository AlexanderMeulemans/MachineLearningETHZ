from sklearn.feature_selection import mutual_info_regression, SelectPercentile
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from imblearn.pipeline import Pipeline
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier,AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, make_scorer
import csv
from imblearn.over_sampling import SMOTE


print('------ opening files -------')
X = np.loadtxt("X5.txt", delimiter=",", dtype="float64")
X_test = np.loadtxt("X_test5.txt", delimiter=",", dtype="float64")
Y = np.loadtxt("Y5.txt", delimiter=",", dtype="float64")


print('------ Preprocess data --------')
imputer = SimpleImputer()
scaler = preprocessing.StandardScaler()

X = imputer.fit_transform(X)
X = scaler.fit_transform(X)


X_test = imputer.transform(X_test)
X_test = scaler.transform(X_test)


print('------ Training classifier with CV -------')

selector = SelectPercentile(mutual_info_regression)

model1 = RandomForestClassifier(n_estimators = 780,class_weight='balanced',min_samples_split=2,min_samples_leaf=4,max_features='sqrt',max_depth=None,bootstrap='False')
model2 = GradientBoostingClassifier()
model3 = GaussianProcessClassifier()
model4 = SVC()
model = VotingClassifier(estimators=[('rfc', model1), ('gb', model2), ('gpc', model3)])
#model = AdaBoostClassifier()

#model = SVC(class_weight='balanced')
pipeline = Pipeline([('MI', selector),
                ('model', model)
                ])

deterministic_grid = {'MI__percentile': [67, 77, 90, 100]}
grid_search_rand = GridSearchCV(pipeline, deterministic_grid, scoring=make_scorer(f1_score, average='micro'), cv=4,
                                verbose=2)
# Fit the model
grid_search_rand.fit(X,Y)
print('best params')
print(grid_search_rand.best_params_)
print("CV results:")
print(grid_search_rand.cv_results_)
