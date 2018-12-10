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

class AlexClassifier(object):

    def __init__(self, depth):
        assert depth >= 1
        self.depth = depth
        self.pipelines = list()
        for _ in range(depth):
            self.pipelines += [
                Pipeline([
                    ('standardizer', preprocessing.StandardScaler()),
                    ('model', skl.RandomForestClassifier(class_weight='balanced'))
                ])
            ]

    def fit(self,X,Y):
        curr_X, depth = X, self.depth
        for i in range(0, self.depth):
            curr_pipeline = self.pipelines[i]
            curr_pipeline.fit(curr_X, Y)
            y_prob = curr_pipeline.predict_proba(curr_X)
            if i != self.depth - 1: curr_X = update_features(X, y_prob)
        return y_prob

    def predict(self,X):
        curr_X, depth = X, self.depth
        for i in range(0, self.depth - 1):
            y_prob = self.pipelines[i].predict_proba(curr_X)
            curr_X = update_features(X, y_prob)

        return self.pipelines[-1].predict(curr_X)

    def predict_proba(self,X):
        curr_X, depth = X, self.depth
        for i in range(0, self.depth - 1):
            y_prob = self.pipelines[i].predict_proba(curr_X)
            curr_X = update_features(X, y_prob)

        return self.pipelines[-1].predict_proba(curr_X)

    def crossvalidate(self,X,Y):
        X_s1 = X[0:21600,:]
        X_s2 = X[21600:43200,:]
        X_s3 = X[43200:,:]
        Y_s1 = Y[0:21600]
        Y_s2 = Y[21600:43200]
        Y_s3 = Y[43200:]
        X_partition = [X_s1, X_s2, X_s3]
        Y_partition = [Y_s1, Y_s2, Y_s3]
        cv_score = []
        for i in range(3):
            X_train = np.concatenate((X_partition[(i+1)%3],X_partition[(i+2)%3]),axis=0)
            print(X_train.shape)
            X_test = X_partition[i]
            print(X_test.shape)
            Y_train = np.concatenate((Y_partition[(i + 1) % 3], Y_partition[(i + 2) % 3]), axis=0)
            Y_test = Y_partition[i]
            self.fit(X_train, Y_train)
            Y_pred = self.predict(X_test)
            score = balanced_accuracy_score(Y_test, Y_pred)
            cv_score.append(score)
        return cv_score

def update_features(X, y_prob):
    pred_delay1 = np.concatenate((np.reshape(y_prob[0,:],(1,-1)),y_prob[0:-1,:]),axis=0)
    pred_delay2 = np.concatenate((np.reshape(y_prob[0,:],(1,-1)), np.reshape(y_prob[0,:],(1,-1)), y_prob[0:-2,:]), axis=0)
    pred_forward1 = np.concatenate((y_prob[1:,:],np.reshape(y_prob[-1,:],(1,-1))),axis=0)
    pred_forward2 = np.concatenate((y_prob[2:,:],np.reshape(y_prob[-1,:],(1,-1)),np.reshape(y_prob[-1,:],(1,-1))),axis=0)
    X_updated = np.concatenate((X, pred_delay2, pred_delay1, pred_forward1, pred_forward2), axis=1)
    return X_updated
