
from sklearn.feature_selection import mutual_info_regression, SelectPercentile
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, make_scorer
from sklearn.impute import SimpleImputer
import sklearn.ensemble as skl
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.preprocessing import LabelEncoder
import imblearn.ensemble as imb
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.pipeline import Pipeline
import xgboost.sklearn as xg
from sklearn.svm import SVC
print('------ opening files -------')
X = np.loadtxt("X5.txt", delimiter=",", dtype="float64")
X_test = np.loadtxt("X_test5.txt", delimiter=",", dtype="float64")
Y = np.loadtxt("Y5.txt", delimiter=",", dtype="float64")

print('------ Training classifier with CV -------')
percentile = 100
imputer = SimpleImputer()
scaler = preprocessing.StandardScaler()
selector = SelectPercentile(mutual_info_regression, percentile=percentile)
<<<<<<< HEAD
over_sample = BorderlineSMOTE()
# 
#model = SVC(class_weight='balanced',gamma = 'scale')
over_sample = SMOTE()
#
#model = SVC(class_weight='balan
#0.668 without any parameters, with 100 estimators 0.709

#model = skl.GradientBoostingClassifier()
#model = xg.XGBClassifier()
model = skl.RandomForestClassifier(class_weight='balanced')
pipeline = Pipeline([
                    ('imputer',imputer),
                    ('standardizer', scaler),
                    #('over_sampler',over_sample),
                    ('MI', selector),
                    ('model',model)
                    ])



# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
print('running rnadom grid search')
random_grid =  {'MI__percentile': [40, 60, 100],
               'model__n_estimators': n_estimators,
               'model__max_features': max_features,
               'model__max_depth': max_depth,
               'model__min_samples_split': min_samples_split,
               'model__min_samples_leaf': min_samples_leaf,
               'model__bootstrap': bootstrap}

<<<<<<< HEAD
grid_search_rand = RandomizedSearchCV(pipeline, random_grid, scoring=make_scorer(f1_score,average='micro'), cv=3, n_iter = 14,verbose=1,n_jobs=3)
grid_search_rand.fit(X,Y)
print('best params')
print(grid_search_rand.best_params_)
#
#
#print("---- predicting ----")
#Y_pred = cross_val_predict(pipeline, X,Y, cv=5)
#
#print("---- scoring ----")
#score = f1_score(Y, Y_pred, average='micro')
#
#print('average CV F1 score: ' + str(score))
=======
print('average CV F1 score: ' + str(score))
>>>>>>> df5a3dcd6fc57d09f5a7e504e658567316a9747a
