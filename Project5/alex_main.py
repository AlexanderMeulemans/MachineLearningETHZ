def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
import csv
from alex_masterplan import AlexClassifier
from alex_pipeline_utils import *

should_preprocess = False
preprocess_dir = "./preprocessed/"

X, X_test = (preprocess_data(preprocess_dir) if
                should_preprocess else load_data(preprocess_dir))

Y = pd.read_csv('train_labels.csv', sep=',', index_col=0)
Y = np.ravel(np.asarray(Y))

print('\n********* Training AlexClassifier')
model_alex = AlexClassifier(depth=3)
model_alex.fit(X,Y)

y_prob = model_alex.predict(X)
score = balanced_accuracy_score(Y, y_prob)
print('average CV F1 score: ' + str(score))

y_pred = model_alex.predict(X_test)

print('\n********* Writing to file')
with open('result.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['Id', 'y'])
    for i in range(len(y_pred)):
        writer.writerow([i, y_pred[i]])
