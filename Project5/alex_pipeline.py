def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

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
import os
from alex_masterplan import AlexClassifier

do_grid_search = False
own_model = True
should_preprocess = False
preprocessed_data_dir = "./preprocessed/"

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

if should_preprocess:
    print('------ preprocessing data -------')
    if not os.path.exists(preprocessed_data_dir):
        os.makedirs(preprocessed_data_dir)

    X = pd.read_csv('train_eeg1.csv', sep=',', index_col=0)
    X = feature_extractor_eeg(np.asarray(X))

    X_test = pd.read_csv('test_eeg1.csv', sep=',', index_col=0)
    X_test = feature_extractor_eeg(np.asarray(X_test))

    np.save(preprocessed_data_dir + "X.npy", X)
    np.save(preprocessed_data_dir + "X_test.npy", X_test)
    print("------- done processing! -------")

else:
    X = np.load(preprocessed_data_dir + "X.npy")
    X_test = np.load(preprocessed_data_dir + "X_test.npy")


Y = pd.read_csv('train_labels.csv', sep=',', index_col=0)
Y = np.asarray(Y)
Y = np.ravel(Y)


print('------ Training Classifier -------')
model_alex = AlexClassifier(depth=5)
model_alex.fit(X,Y)

y_prob = model_alex.predict(X)
score = balanced_accuracy_score(Y, y_prob)
print('average CV F1 score: ' + str(score))

y_pred = model_alex.predict(X_test)


# print("----- crossvalidating ----- ")
# cv = KFold(n_splits=3, shuffle=False)
# Y_pred = cross_val_predict(pipeline, X, Y, cv=cv, verbose=1)
# score = balanced_accuracy_score(Y, Y_pred)
#
# print('average CV F1 score: ' + str(score))
#
# print('---- Run on test data -------')
# model = skl.RandomForestClassifier(class_weight='balanced', n_estimators=100)
# pipeline = Pipeline([
#     ('standardizer', scaler),
#     ('model', model)
# ])
# pipeline.fit(X, Y)
# y_pred = pipeline.predict(X_test)

print('writing to file')
with open('result.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['Id', 'y'])
    for i in range(len(y_pred)):
        writer.writerow([i, y_pred[i]])
