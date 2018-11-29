from get_data import get_videos_from_folder,get_target_from_csv
import os
import numpy as np
from utils import save_solution
import cv2 as cv
import matplotlib.pyplot as plt
import math as math

dir_path = os.path.dirname(os.path.realpath(__file__))
train_folder = os.path.join(dir_path,"train")
test_folder = os.path.join(dir_path,"test")

train_target = os.path.join(dir_path,'train_target.csv')
my_solution_file = os.path.join(dir_path,'karpsolution.csv')

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')

bgf= cv.createBackgroundSubtractorMOG2()
#If you want to look at the hearts
plt.show()
for j, vid in enumerate(x_train):
    filtervid = np.zeros(np.shape(vid))
    for i in range(len(vid[:,0,0])):
        frame = vid[i,:,:]
        mask = bgf.apply(frame)
#        If you want to see videos, uncomment
#        cv.imshow('frame',mask)
#        cv.imshow('frame2',frame)
#        cv.waitKey(30)
        filtervid[i,:,:]=mask
    x_train[j]=filtervid

for j, vid in enumerate(x_test):
    filtervid = np.zeros(np.shape(vid))
    for i in range(len(vid[:,0,0])):
        frame = vid[i,:,:]
        mask = bgf.apply(frame)
        filtervid[i,:,:]=mask
    x_test[j]=filtervid
#%% 
heartbeats=np.zeros(len(x_train))
for k, vid in enumerate(x_train):
    x = np.zeros(len(np.fft.rfft(vid[:,1,1])))
    for i in range(len(vid[1,:,1])):
        for j in range(len(vid[1,1,:])):
            xp = np.fft.rfft(vid[:,i,j])
            x = x + xp
    x = x / np.sum(x)
    xf = np.fft.rfftfreq(len(vid[:,1,1]),1/25)
    plot(xf,x)
    heartbeats[k] = xf[np.argmax(x))]
#%%
plt.plot(heartbeats)
plt.show()
    

#last step
#save_solution(my_solution_file,solution)
