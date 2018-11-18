
from sklearn.feature_selection import mutual_info_regression, SelectPercentile
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, make_scorer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

import numpy as np
from sklearn.preprocessing import LabelEncoder

print('------ opening files -------')
X = np.loadtxt("X3.txt", delimiter=",", dtype="float64")
X_test = np.loadtxt("X_test3.txt", delimiter=",", dtype="float64")
Y = np.loadtxt("Y3.txt", delimiter=",", dtype="float64")
print(X.shape)

print('------ Training classifier with CV -------')
percentile = 100
imputer = SimpleImputer()
scaler = preprocessing.StandardScaler()
selector = SelectPercentile(mutual_info_regression, percentile=percentile)

#
#model = SVC(class_weight='balanced')
#%%
#0.668 without any parameters, with 100 estimators 0.709
model = RandomForestClassifier(n_estimators = 100,class_weight='balanced')

pipeline = Pipeline([
                    ('imputer',imputer),
                    ('standardizer', scaler),
                    ('MI', selector),
                    ('model',model)
                    ])



# Number of trees in random forest
n_estimators = [1000,1500,1800,1900,2000]
# Number of features to consider at every split
max_features = ['auto']
# Maximum number of levels in tree
max_depth = [50]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [8,10,12]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [False]
# Percentile feature selector
percentile = [x for x in np.linspace(50, 100, num = 22)]
# Create the random grid
print('running rnadom grid search')
random_grid =  {'MI__percentile': percentile,
               'model__n_estimators': n_estimators,
               'model__max_features': max_features,
               'model__max_depth': max_depth,
               'model__min_samples_split': min_samples_split,
               'model__min_samples_leaf': min_samples_leaf,
               'model__bootstrap': bootstrap}

grid_search_rand = RandomizedSearchCV(pipeline, random_grid,
                                      scoring=make_scorer(f1_score,
                                                          average='micro'),
                                      cv=8, n_iter = 30,verbose=2)
grid_search_rand.fit(X,Y)
print('best params')
print(grid_search_rand.best_params_)
print("CV results:")
print(grid_search_rand.cv_results_)


#print("---- predicting ----")
#Y_pred = cross_val_predict(pipeline, X,Y, cv=5)
#
#print("---- scoring ----")
#score = f1_score(Y, Y_pred, average='micro')
#
#print('average CV F1 score: ' + str(score))