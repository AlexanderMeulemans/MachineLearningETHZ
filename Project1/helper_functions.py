
from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneOut, cross_val_predict, permutation_test_score
from scipy import stats
from sklearn.feature_selection import VarianceThreshold, f_regression, mutual_info_regression, SelectPercentile
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import r2_score


def cross_val_output(features,labels,pipeline,title,name_outputfile,cv = None, scoring = 'r2'):
    """
    Do a cross validation (LOOCV) on the samples and write the results to the result text file
    :param features: np.array of features of all the samples (2D array with as rows the samples and as columns the features)
    :param labels: lables of the samples
    :param pipeline: pipeline of feature selector + regressor
    :param title: name of the result that will be displayed as title in the text file
    :param cv: Cross validation strategy to be applied. If None, 5 kfold is used
    :param permutations: number of permutation tests to be executed
    """
    if cv is None:
        cv = 5
    labels_pred = cross_val_predict(pipeline, features, labels, cv=cv)
    score = r2_score(labels, labels_pred)
    outputfile = open(name_outputfile,'a')
    outputfile.write('%s _______________________________________\n' %title)
    outputfile.write('R^2 score: %s \n' %str(score))
    outputfile.close()

def grid_search_output(X, y, pipeline, parameterset, title, name_outputfile, cv = 5, scoring = 'r2', n_iter = 10):
    """
    Do a grid search over parameterset and write best results to outputfile
    :param features:
    :param labels:
    :param pipeline:
    :param parameterset:
    :param title:
    :param name_outputfile:
    :param cv:
    :param scoring:
    :return:
    """

    search = RandomizedSearchCV(pipeline, parameterset, scoring=scoring, n_iter=n_iter, cv=cv)
    search.fit(X, y)
    f = open(name_outputfile, 'a')
    f.write('Gridsearch ' + title + ' ________________ \n')
    f.write("Best parameter (CV score=%0.3f):" % search.best_score_)
    f.write(str(search.best_params_))
    f.write('detailed cross validation results: ')
    f.write(str(search.cv_results_))
    f.write('\n')
    f.close()
    return search
