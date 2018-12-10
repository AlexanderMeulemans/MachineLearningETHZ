from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from alex_pipeline_utils import *
import csv

should_preprocess = True
preprocess_dir = "./preprocessed/"

X, X_test = (preprocess_all_data(preprocess_dir) if
               should_preprocess else load_data(preprocess_dir, "all"))
Y = np.ravel(np.asarray(pd.read_csv('train_labels.csv', sep=',', index_col=0)))

print('------ Training classifier with CV -------')
# model = skl.RandomForestClassifier(class_weight='balanced',n_estimators=100)

model = SVC(class_weight='balanced')
model = Pipeline([('standardizer', preprocessing.StandardScaler()),
                    ('model',model)
                    ])

cv = KFold(n_splits=3,shuffle=False)
Y_pred = cross_val_predict(model, X, Y, cv=cv)
score = balanced_accuracy_score(Y, Y_pred)
print('balanced accuracy score: ' + str(score))

print('\n********* Writing to file')
model.fit(X,Y)
y_pred=model.predict(X_test)
#%%
with open('result.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['Id', 'y'])
    for i in range(len(y_pred)):
        writer.writerow([i, y_pred[i]])
