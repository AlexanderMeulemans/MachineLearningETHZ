from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from alex_pipeline_utils import *
import sklearn.ensemble as skl
import csv
import matplotlib.pyplot as plt
import numpy as np


should_preprocess = False
should_add_before_after = True
preprocess_dir = "./preprocessed/"

X, X_test = (preprocess_all_data(preprocess_dir) if
               should_preprocess else load_data(preprocess_dir, "all"))
Y = np.ravel(np.asarray(pd.read_csv('train_labels.csv', sep=',', index_col=0)))

#uncomment if you wanna load eeg1 data instead
#X, X_test = (preprocess_data(preprocess_dir, "eeg1") if
#                should_preprocess else load_data(preprocess_dir, "eeg1"))
#Y = np.ravel(np.asarray(pd.read_csv('train_labels.csv', sep=',', index_col=0)))


if should_add_before_after:
    X1 = np.zeros(X.shape)
    X1[0,:] = 3*X[0,:]
    X1[-1,:] = 3*X[-1,:]
    for i in range(1,len(X)-1):
        X1[i,:] = X[i-1,:] + X[i,:] + X[i+1,:]
    X = X1
    
    
    X1 = np.zeros(X_test.shape)
    X1[0,:] = 3*X_test[0,:]
    X1[-1,:] = 3*X_test[-1,:]
    for i in range(1,len(X_test)-1):
        X1[i,:] = X_test[i-1,:] + X_test[i,:] + X_test[i+1,:]
    X_test = X1

#%%
print('------ Training classifier with CV -------')
#model = skl.RandomForestClassifier(class_weight='balanced',n_estimators=100)

model = SVC(class_weight='balanced')
model = Pipeline([('standardizer', preprocessing.StandardScaler()),
                    ('model',model)
                    ])

cv = KFold(n_splits=3,shuffle=False)
Y_pred = cross_val_predict(model, X, Y, cv=cv)
score = balanced_accuracy_score(Y, Y_pred)
print('balanced accuracy score: ' + str(score))

model.fit(X,Y)
#%%
plt.figure(1)
plt.plot(Y_pred,'x')
y_pred=model.predict(X_test)
plt.figure(2)
plt.plot(y_pred,'x')
#%%

print('\n********* Writing to file')
with open('result.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['Id', 'y'])
    for i in range(len(y_pred)):
        writer.writerow([i, y_pred[i]])
