import tensorflow as tf

class Dataset:
	def __init__(self):
		pass

	def __preprocess__(self):
		self.trainX = self.trainX.astype('float32')
		self.testX = self.testX.astype('float32')
		self.trainX = self.trainX * 1. / 255.0
		self.testX = self.testX * 1. / 255.0