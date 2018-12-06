
from sklearn.feature_selection import mutual_info_regression, SelectPercentile
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import f1_score, make_scorer, balanced_accuracy_score
from sklearn.impute import SimpleImputer
import sklearn.ensemble as skl
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import imblearn.ensemble as imb
from imblearn.pipeline import Pipeline
import xgboost.sklearn as xg
from sklearn.svm import SVC
import pandas as pd


def grid_treepipe_search():
    
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
    
    grid_search_rand = RandomizedSearchCV(pipeline, random_grid, scoring=make_scorer(f1_score,average='micro'), cv=3, n_iter = 14,verbose=1,n_jobs=3)
    return grid_search_rand



print('------ opening files -------')
X = pd.read_csv('train_eeg1.csv',sep=',',index_col=0)
X = np.asarray(X)
X = np.fft.fft(X)
X = np.abs(X)
#%%
Y = pd.read_csv('train_labels.csv',sep=',',index_col=0)
Y = np.asarray(Y)
Y = np.ravel(Y)

X_test = pd.read_csv('test_eeg1.csv',sep=',',index_col=0)
X_test = np.asarray(X_test)
X_test = np.fft.fft(X_test)
X_test = np.abs(X_test)

print('------ Training classifier with CV -------')
percentile = 100
imputer = SimpleImputer()
scaler = preprocessing.StandardScaler()
selector = SelectPercentile(mutual_info_regression, percentile=percentile)


model = skl.RandomForestClassifier(class_weight='balanced')
pipeline = Pipeline([
                    ('imputer',imputer),
                    ('standardizer', scaler),
                    ('MI', selector),
                    ('model',model)
                    ])
    
cv = KFold(n_splits=3,shuffle=False)
Y_pred = cross_val_predict(model, X, Y, cv=cv)
score = balanced_accuracy_score(Y, Y_pred)

print('average CV F1 score: ' + str(score))
