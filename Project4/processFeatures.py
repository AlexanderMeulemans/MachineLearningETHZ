import os
import numpy as np
from tf_utils import input_fn_from_dataset,input_fn_frame_from_dataset,save_tf_record,prob_positive_class_from_prediction
from get_data import get_videos_from_folder,get_target_from_csv, get_full_data
from utils import save_solution

from sklearn import preprocessing

print('------ Loading data ------')
dir_path = os.path.dirname(os.path.realpath(__file__))
train_folder = os.path.join(dir_path,"./train/")
test_folder = os.path.join(dir_path,"./test/")

train_target = os.path.join(dir_path,'./train_target.csv')
my_solution_file = os.path.join(dir_path,'./solution.csv')

X_train, y = get_full_data(train_folder, train_target)
X_test = get_videos_from_folder(test_folder)


print('----------- processing data -----------')
print(X_train.shape)
print(y.shape)
print(X_test.shape)

scaler = preprocessing.StandardScaler()
X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3]))
X_train_reshaped = scaler.fit_transform(X_train_reshaped)
X_train = np.reshape(X_train_reshaped, X_train.shape)

np.save("X.npy", X_train)
np.save("X_test.npy", X_test)
np.save("Y.npy", y)
