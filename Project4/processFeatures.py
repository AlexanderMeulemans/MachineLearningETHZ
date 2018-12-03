import os
import numpy as np
from tf_utils import input_fn_from_dataset,input_fn_frame_from_dataset,save_tf_record,prob_positive_class_from_prediction
from get_data import get_videos_from_folder,get_target_from_csv
from utils import save_solution

from sklearn import preprocessing

print('------ Loading data ------')
dir_path = os.path.dirname(os.path.realpath(__file__))
train_folder = os.path.join(dir_path,"./train/")
test_folder = os.path.join(dir_path,"./test/")

train_target = os.path.join(dir_path,'./train_target.csv')
my_solution_file = os.path.join(dir_path,'./solution.csv')

y = get_target_from_csv(train_target)
X_train = get_videos_from_folder(train_folder)
X_test = get_videos_from_folder(test_folder)

print('----------- processing data -----------')
scaler = preprocessing.StandardScaler()
X_train_reshaped = np.reshape(X_train, (158, 220000))
X_train_reshaped = scaler.fit_transform(X_train_reshaped)
X_train = np.reshape(X_train_reshaped, X_train.shape)

np.save("X.npy", X_train)
np.save("X_test.npy", X_test)
np.save("Y.npy", y)
