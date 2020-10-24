import tensorflow_datasets as tfds 
import tensorflow as tf
try:
    from Dataset import Dataset
except ImportError:
    from .Dataset import Dataset

class ImageNet(Dataset):
	def __init__(self):
		ds = tfds.as_numpy(tfds.load('imagenet_resized/32x32', batch_size=-1))

		self.trainX, self.trainY = ds["train"]["image"], ds["train"]["label"]
		self.testX, self.testY = ds["validation"]["image"], ds["validation"]["label"]

		self.trainY = tf.keras.utils.to_categorical(self.trainY)
		self.testY = tf.keras.utils.to_categorical(self.testY)

		self.__preprocess__()

if __name__ == '__main__':
	ImageNet = ImageNet()
	print(ImageNet.trainX.shape)
	print(ImageNet.trainY.shape)
	print(ImageNet.testX.shape)
	print(ImageNet.testY.shape)