from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, balanced_accuracy_score

from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd
import os
from alex_featureselection import feature_extractor_eeg


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
    print('\n********* Performing Grid Search')
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


def preprocess_data(preprocessed_data_dir, data_type):
    print('\n********* Preprocessing Data')
    if not os.path.exists(preprocessed_data_dir):
        os.makedirs(preprocessed_data_dir)

    X = pd.read_csv('train_' + data_type + '.csv', sep=',', index_col=0)
    X = feature_extractor_eeg(np.asarray(X))

    X_test = pd.read_csv('test_' + data_type + '.csv', sep=',', index_col=0)
    X_test = feature_extractor_eeg(np.asarray(X_test))

    np.save(preprocessed_data_dir + "X_" + data_type + ".npy", X)
    np.save(preprocessed_data_dir + "X_" + data_type +"_test.npy", X_test)
    print('\n********* Done Processing')
    return X, X_test


def load_data(preprocessed_data_dir, data_type):
    X = np.load(preprocessed_data_dir + "X_" + data_type + ".npy")
    X_test = np.load(preprocessed_data_dir + "X_" + data_type +"_test.npy")
    return X, X_test
