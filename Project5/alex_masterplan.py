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

    def __init__(self, scaler):
        model = skl.RandomForestClassifier(class_weight='balanced')
        pipeline = Pipeline([
            ('standardizer', scaler),
            ('model', model)
        ])
        self.pipeline_base = pipeline

        model2 = skl.RandomForestClassifier(class_weight='balanced')
        self.pipeline_phase1 = Pipeline([
            ('standardizer', scaler),
            ('model', model2)
        ])

    def fit(self,X,Y):
        self.pipeline_base.fit(X,Y)
        Y_logprob = self.pipeline_base.predict_proba(X)
        X_updated = update_features(X,Y_logprob)
        self.pipeline_phase1.fit(X_updated, Y)

    def predict(self,X):
        Y_prob = []
        for x in X:
            Y_prob += [self.pipeline_base.predict_proba(X)]
        X_updated = update_features(X, Y_prob)
        Y_pred = self.pipeline_phase1.predict(X_updated)
        return Y_pred

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

def update_features(X,y_prob):
    pred_delay1 = np.concatenate((np.reshape(y_prob[0,:],(1,-1)),y_prob[0:-1,:]),axis=0)
    pred_delay2 = np.concatenate((np.reshape(y_prob[0,:],(1,-1)), np.reshape(y_prob[0,:],(1,-1)), y_prob[0:-2,:]), axis=0)
    pred_forward1 = np.concatenate((y_prob[1:,:],np.reshape(y_prob[-1,:],(1,-1))),axis=0)
    pred_forward2 = np.concatenate((y_prob[2:,:],np.reshape(y_prob[-1,:],(1,-1)),np.reshape(y_prob[-1,:],(1,-1))),axis=0)
    X_updated = np.concatenate((X, pred_delay2, pred_delay1, pred_forward1, pred_forward2), axis=1)
    return X_updated
