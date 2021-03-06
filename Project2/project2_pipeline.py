import numpy as np
from sklearn.feature_selection import mutual_info_regression, SelectPercentile
from sklearn import preprocessing
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy import stats
# from sklearn.utils import stats

percentile = 40

#%%

print("OPENING FILES")
Y = np.genfromtxt("y_train.csv", delimiter=",", skip_header=1)
X = np.genfromtxt("X_train.csv", delimiter=",", skip_header=1)

Y = np.ravel(Y[:, 1:])
X = np.array(X[:, 1:])

#%%
scaler = preprocessing.StandardScaler()
svc = SVC(kernel='rbf', class_weight='balanced')

pipeline = Pipeline([('standardizer', scaler),
                     ('MI', SelectPercentile(mutual_info_regression)),
                     ('svc', svc)])
#
param_grid = {'MI__percentile': [20, 40, 60, 80], 'svc__kernel': ['rbf', 'poly', 'sigmoid']}
param_grid_random = {'MI__percentile': stats.beta(a=4,b=2, scale=100, loc=0),
    'svc__C': stats.gamma(a=2,scale=50),
    'svc__gamma': stats.gamma(a=2,scale=1/(X.shape[1]*0.5))}

print("STARTING GRID SEARCH")
# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=make_scorer(balanced_accuracy_score))
# grid_search.fit(X, Y)

grid_search_rand = RandomizedSearchCV(pipeline, param_grid_random, scoring=make_scorer(balanced_accuracy_score), cv=5, n_iter = 12)
grid_search_rand.fit(X,Y)

print("DONE BITCH. BEST PARAMS")
# print(grid_search.best_params_)
print(grid_search_rand.best_params_)

# predictions = grid_search.predict(X)
predictions = grid_search_rand.predict(X)
score = balanced_accuracy_score(Y, predictions)
print('Balanced Accuracy Score: ' + str(score))

