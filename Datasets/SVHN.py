import tensorflow_datasets as tfds 
import tensorflow as tf
try:
    from Dataset import Dataset
except ImportError:
    from .Dataset import Dataset

class SVHN(Dataset):
	def __init__(self):
		ds = tfds.as_numpy(tfds.load('svhn_cropped', batch_size=-1))

		self.trainX, self.trainY = ds["train"]["image"], ds["train"]["label"]
		self.testX, self.testY = ds["test"]["image"], ds["test"]["label"]

		self.trainY = tf.keras.utils.to_categorical(self.trainY)
		self.testY = tf.keras.utils.to_categorical(self.testY)

		self.__preprocess__()
		self.classLabels=None

if __name__ == '__main__':
	SVHN = SVHN()
	print(SVHN.trainX.shape)
	print(SVHN.trainY.shape)
	print(SVHN.testX.shape)
	print(SVHN.testY.shape)