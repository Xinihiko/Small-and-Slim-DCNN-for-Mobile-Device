import numpy as np
try:
    from Dataset import Dataset
except ImportError:
    from .Dataset import Dataset
import os

class GTSRB(Dataset):
	def __init__(self):
		print(os.path.abspath(__file__))
		datasetloc = os.path.abspath(__file__).split('\\')[:-1]
		datasetloc = "\\".join(datasetloc)
		# load the dataset that have been converted to npz file using GTSRB_to_npz.py
		dataset = np.load(datasetloc+'\\GTSRB-train.npz')
		self.trainX = dataset['images']
		self.trainY = dataset['labels']

		dataset = np.load(datasetloc+'\\GTSRB-test.npz')
		self.testX = dataset['images']
		self.testY = dataset['labels']

		self.__preprocess__()
		self.classLabels=None

if __name__ == '__main__':
	GTSRB = GTSRB()
	print(GTSRB.trainX.shape)
	print(GTSRB.trainY.shape)
	print(GTSRB.testX.shape)
	print(GTSRB.testY.shape)

	# print(GTSRB.trainX[0])