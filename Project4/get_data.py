import numpy as np
import skvideo
from sklearn import preprocessing

skvideo.setFFmpegPath("/usr/local/bin")
import skvideo.io
import os 
import sys 
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))
def get_videos_from_folder(data_folder):
	data_folder = os.path.join(dir_path,data_folder)
	file_names, x = [], []
	scaler = preprocessing.StandardScaler()

	if os.path.isdir(data_folder):
		for dirpath, dirnames, filenames in os.walk(data_folder):
			for filename in filenames:
				file_path = os.path.join(dirpath, filename)
				statinfo = os.stat(file_path)
				if statinfo.st_size != 0:
					video = skvideo.io.vread(file_path, outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
					video = np.expand_dims(video[:22,:,:], axis=3)
					x.append(video)
					file_names.append(int(filename.split(".")[0]))
	return np.asarray(x)

def get_target_from_csv(csv_file):
	csv_file = os.path.join(dir_path, csv_file)
	with open(csv_file, 'r') as csvfile:
		label_reader = pd.read_csv(csvfile)
		y = label_reader['y']
	y = np.array(y)
	return y