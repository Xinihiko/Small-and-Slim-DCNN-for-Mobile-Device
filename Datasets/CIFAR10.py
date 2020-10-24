from tensorflow.keras.datasets import cifar10
import tensorflow as tf
try:
    from Dataset import Dataset
except ImportError:
    from .Dataset import Dataset

class CIFAR10(Dataset):
	def __init__(self):
		(self.trainX, self.trainY), (self.testX, self.testY) = cifar10.load_data()

		self.trainY = tf.keras.utils.to_categorical(self.trainY)
		self.testY = tf.keras.utils.to_categorical(self.testY)

		self.__preprocess__()

		self.classLabels = ["airplane", "automobile", "bird", "cat", "deer",
						"dog", "frog", "horse", "ship", "truck"]

if __name__ == '__main__':
	CIFAR10 = CIFAR10()
	print(CIFAR10.trainX.shape)
	print(CIFAR10.trainY.shape)
	print(CIFAR10.testX.shape)
	print(CIFAR10.testY.shape)

	print(CIFAR10.trainY[0])