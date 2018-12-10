def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import pandas as pd
import csv
from allan_classifier import AllanClassifier
from alex_pipeline_utils import *

should_preprocess = False
preprocess_dir = "./preprocessed/"

X1, X1_test = (preprocess_data(preprocess_dir, "eeg1") if
                should_preprocess else load_data(preprocess_dir, "eeg1"))
X2, X2_test = (preprocess_data(preprocess_dir, "eeg2") if
                should_preprocess else load_data(preprocess_dir, "eeg2"))

Y = np.ravel(np.asarray(pd.read_csv('train_labels.csv', sep=',', index_col=0)))

print('\n********* Training AllanClassifier')
model = AllanClassifier(depth=3)
model.fit(X1, X2, Y)

y_prob = model.predict(X1, X2)
print(y_prob)
score = balanced_accuracy_score(Y, y_prob)
print('average CV F1 score: ' + str(score))

y_pred = model.predict(X1_test, X2_test)

print('\n********* Writing to file')
with open('result.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['Id', 'y'])
    for i in range(len(y_pred)):
        writer.writerow([i, y_pred[i]])