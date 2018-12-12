import numpy as np
import pandas as pd
from alex_featureselection import total_feature_extractor
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, KFold
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from imblearn.pipeline import Pipeline
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import GridSearchCV
from alex_classifier import AlexClassifier

import sklearn.ensemble as skl
import csv

# Pipeline variables:
should_add_before_after = False
should_run_on_test = False
test_different_models = False
do_gridsearch = False
own_model = False
voting_classifier = True


print('------ opening files -------')
X_eeg1 = pd.read_csv('train_eeg1.csv',sep=',',index_col=0)
X_eeg1 = np.asarray(X_eeg1)


X_eeg2 = pd.read_csv('train_eeg2.csv',sep=',',index_col=0)
X_emg = np.asarray(X_eeg2)

X_emg = pd.read_csv('train_emg.csv',sep=',',index_col=0)
X_emg = np.asarray(X_emg)


#%%
Y = pd.read_csv('train_labels.csv',sep=',',index_col=0)
Y = np.asarray(Y)
Y = np.ravel(Y)

X_test_eeg1 = pd.read_csv('test_eeg1.csv',sep=',',index_col=0)
X_test_eeg1 = np.asarray(X_test_eeg1)

X_test_eeg2 = pd.read_csv('test_eeg2.csv',sep=',',index_col=0)
X_test_eeg2 = np.asarray(X_test_eeg2)

X_test_emg = pd.read_csv('test_emg.csv',sep=',',index_col=0)
X_test_emg = np.asarray(X_test_emg)


print('------ preprocessing files ------')
X = total_feature_extractor(X_eeg1,X_eeg2, X_emg)
X_test = total_feature_extractor(X_test_eeg1,X_test_eeg2, X_test_emg)

if should_add_before_after:
    X1 = np.zeros(X.shape)
    X1[0, :] = 3 * X[0, :]
    X1[-1, :] = 3 * X[-1, :]
    for i in range(1, len(X) - 1):
        X1[i, :] = X[i - 1, :] + X[i, :] + X[i + 1, :]
    X = X1

    X1 = np.zeros(X_test.shape)
    X1[0, :] = 3 * X_test[0, :]
    X1[-1, :] = 3 * X_test[-1, :]
    for i in range(1, len(X_test) - 1):
        X1[i, :] = X_test[i - 1, :] + X_test[i, :] + X_test[i + 1, :]
    X_test = X1

print('----- Training models ------')

if voting_classifier:
    model1 = LinearSVC(class_weight='balanced', max_iter=3000, dual=False)
    model2 = SVC(class_weight='balanced')
    model3 = BalancedBaggingClassifier(base_estimator=model2, n_estimators=100)
    model = skl.VotingClassifier([('LinearSVC', model1), ('SVC', model2), ('balancedbagging', model3)])
    model = Pipeline([('standardizer', preprocessing.StandardScaler()),
                      ('model', model)
                      ])
    scorer = make_scorer(balanced_accuracy_score)
    cv = KFold(n_splits=3, shuffle=False)
    score = cross_val_score(model, X, Y, cv=cv, scoring=scorer)
    print('voting classifier: {}'.format(score))

if own_model:
    model = AlexClassifier(2)
    cv_score = model.crossvalidate(X,Y)
    print('own model: {}'.format(cv_score))


if test_different_models:
    model1 = LinearSVC(class_weight='balanced',max_iter=3000,dual=False)
    model2 = SVC(class_weight='balanced')
    model3 = LogisticRegression(class_weight='balanced',solver='lbfgs',multi_class='auto')
    model4 = RandomForestClassifier(n_estimators=100, class_weight='balanced')
    model5 = BalancedBaggingClassifier(base_estimator = model2, n_estimators=100)
    model_list = [model1,model2,model3,model4,model5]

    for idx, model in enumerate(model_list):
        selector = SelectPercentile(mutual_info_classif,90)
        model = Pipeline([('standardizer', preprocessing.StandardScaler()),
                          ('MI', selector),
                            ('model', model)
                            ])
        scorer = make_scorer(balanced_accuracy_score)
        cv = KFold(n_splits=3,shuffle=False)
        score = cross_val_score(model, X, Y, cv=cv, scoring=scorer)
        print('balanced accuracy score model {}: {}'.format(idx+1, score))

if do_gridsearch:
    selector = SelectPercentile(mutual_info_classif, 90)
    model_svc = SVC(class_weight='balanced')
    model = BalancedBaggingClassifier(base_estimator=model_svc, n_estimators=100)
    pipeline = Pipeline([('standardizer', preprocessing.StandardScaler()),
                      ('MI', selector),
                      ('model', model)
                      ])
    percentiles = [70, 80, 90, 100]
    # Create the random grid
    print('\n********* Performing Grid Search')
    random_grid = {'MI__percentile': percentiles}

    grid_search_rand = GridSearchCV(pipeline, random_grid, scoring=make_scorer(balanced_accuracy_score),
                                          cv=KFold(n_splits=3, shuffle=False),
                                        verbose=1, n_jobs=3)
    grid_search_rand.fit(X, Y)
    print("DONE BITCH. BEST PARAMS")
    # print(grid_search.best_params_)
    print(grid_search_rand.best_params_)
    print("Results:")
    print('Best score: {}'.format(grid_search_rand.best_score_))
    print(grid_search_rand.cv_results_)

if should_run_on_test:
    selector = SelectPercentile(mutual_info_classif, 90)
    model_svc = SVC(class_weight='balanced')
    model = BalancedBaggingClassifier(base_estimator=model_svc, n_estimators=100)
    pipeline = Pipeline([('standardizer', preprocessing.StandardScaler()),
                         ('model', model)
                         ])
    pipeline.fit(X,Y)
    #%%
    y_pred = pipeline.predict(X_test)

    print('\n********* Writing to file')
    with open('result.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['Id', 'y'])
        for i in range(len(y_pred)):
            writer.writerow([i, y_pred[i]])




