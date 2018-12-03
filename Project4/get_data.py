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
	data_folder = os.path.join(dir_path, data_folder)
	file_names, x, y_copies = [], [], []

	if os.path.isdir(data_folder):
		for dirpath, dirnames, filenames in os.walk(data_folder):
			for filename in filenames:
				file_path = os.path.join(dirpath, filename)
				statinfo = os.stat(file_path)
				if statinfo.st_size != 0:
					video = skvideo.io.vread(file_path, outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
					length = video.shape[0]
					number_of_splits = length // 20
					y_copies += [number_of_splits]
					for i in range(number_of_splits):
						start = i * 20
						x.append(np.expand_dims(video[start: start + 20,:,:], axis=3))
					file_names.append(int(filename.split(".")[0]))

	return np.asarray(x), y_copies

def get_full_data(X_folder, y_file):
	x, y_copies = get_videos_from_folder(X_folder)
	basis_y, processed_y = get_target_from_csv(y_file), []
	for i in range(len(basis_y)):
		processed_y += [basis_y[i]] * y_copies[i]

	return x, np.asarray(processed_y)

def get_target_from_csv(csv_file):
	csv_file = os.path.join(dir_path, csv_file)
	with open(csv_file, 'r') as csvfile:
		label_reader = pd.read_csv(csvfile)
		y = label_reader['y']
	y = np.array(y)
	return y
