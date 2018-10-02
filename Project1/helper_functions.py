
from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneOut, cross_val_predict, permutation_test_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import r2_score


def cross_val_output(features,labels,pipeline,title,name_outputfile,cv = None):
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

