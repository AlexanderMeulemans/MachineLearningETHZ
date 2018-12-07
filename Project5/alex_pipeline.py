from sklearn.feature_selection import mutual_info_regression, SelectPercentile
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import f1_score, make_scorer, balanced_accuracy_score
from sklearn.impute import SimpleImputer
import sklearn.ensemble as skl
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import imblearn.ensemble as imb
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pandas as pd
from alex_featureselection import feature_extractor_eeg
import csv
from alex_masterplan import AlexClassifier

do_grid_search = False
own_model = True

def grid_treepipe_search(pipeline,X,Y):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.logspace(start=1, stop=2.3, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    percentiles = [80,90,100]
    # Create the random grid
    print('running random grid search')
    random_grid = {'MI__percentile': percentiles,
                   'model__n_estimators': n_estimators,
                   'model__max_features': max_features,
                   'model__max_depth': max_depth,
                   'model__min_samples_split': min_samples_split,
                   'model__min_samples_leaf': min_samples_leaf,
                   'model__bootstrap': bootstrap}

    grid_search_rand = RandomizedSearchCV(pipeline, random_grid, scoring=make_scorer(balanced_accuracy_score),
                                          cv=KFold(n_splits=3, shuffle=False),
                                          n_iter=10, verbose=1, n_jobs=3)
    grid_search_rand.fit(X, Y)
    print("DONE BITCH. BEST PARAMS")
    # print(grid_search.best_params_)
    print(grid_search_rand.best_params_)
    print("Results:")
    print('Best score: {}'.format(grid_search_rand.best_score_))
    print(grid_search_rand.cv_results_)
    return grid_search_rand


print('------ opening files -------')
X = pd.read_csv('train_eeg1.csv', sep=',', index_col=0)
X = np.asarray(X)
X = feature_extractor_eeg(X)
# %%
Y = pd.read_csv('train_labels.csv', sep=',', index_col=0)
Y = np.asarray(Y)
Y = np.ravel(Y)

X_test = pd.read_csv('test_eeg1.csv', sep=',', index_col=0)
X_test = np.asarray(X_test)
X_test = feature_extractor_eeg(X_test)

print('------ Training classifier with CV -------')
percentile = 100
imputer = SimpleImputer()
scaler = preprocessing.StandardScaler()
selector = SelectPercentile(mutual_info_regression, percentile=percentile)

model = skl.RandomForestClassifier(class_weight='balanced')
pipeline = Pipeline([
    #('imputer', imputer),
    ('standardizer', scaler),
    ('MI', selector),
    ('model', model)
])

if do_grid_search:
    results = grid_treepipe_search(pipeline,X,Y)
else:
    if own_model:
        model_alex = AlexClassifier(scaler)
        # cv_results = model_alex.crossvalidate(X,Y)
        # print('CV results: {}'.format(cv_results))
        print('---- run on test data -----')
        model_alex.fit(X,Y)
        y_pred = model_alex.predict(X_test)

    else:
        cv = KFold(n_splits=3, shuffle=False)
        Y_pred = cross_val_predict(pipeline, X, Y, cv=cv)
        score = balanced_accuracy_score(Y, Y_pred)

        print('average CV F1 score: ' + str(score))

        print('---- Run on test data -------')
        model = skl.RandomForestClassifier(class_weight='balanced', n_estimators=100)
        pipeline = Pipeline([
            ('standardizer', scaler),
            ('model', model)
        ])
        pipeline.fit(X, Y)
        y_pred = pipeline.predict(X_test)

    print('writing to file')
    with open('result.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['Id', 'y'])
        for i in range(len(y_pred)):
            writer.writerow([i, y_pred[i]])





