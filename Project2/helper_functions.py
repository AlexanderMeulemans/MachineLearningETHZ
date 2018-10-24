import numpy as np


from sklearn.model_selection import cross_val_predict

from sklearn.metrics import balanced_accuracy_score

def class2sample_weights(y,class_weights):
    sample_weights = []
    for yi in y:
        sample_weights.append(class_weights[yi])
    return np.array(sample_weights)

def clip_resultfile(y_pred):
    result_file = []
    for yi in y_pred:
        label = np.argmax(yi)
        result_file.append(label)
    return np.array(result_file)




def cross_val_output(features,labels,pipeline,title,name_outputfile=None,cv = None, scoring = 'r2'):
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
    score = balanced_accuracy_score(labels, labels_pred)

    if name_outputfile:
        outputfile = open(name_outputfile,'a')
        outputfile.write('%s _______________________________________\n' %title)
        outputfile.write('Balanced Accuracy Score: ' + str(score))
        outputfile.close()
    else:
        print('Balanced Accuracy Score: ' + str(score))
