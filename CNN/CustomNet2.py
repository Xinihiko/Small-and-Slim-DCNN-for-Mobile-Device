import tensorflow as tf
try:
    from Model import Model
except ImportError:
    from .Model import Model

class CustomNet2(Model):
	def __init__(self, input_size, output_size, init_lr=1e-1, epoch=50, batch=32, optimizers='SGD', load_path=""):
		super().__init__(input_size, output_size, init_lr, epoch, batch, optimizers, load_path)
		self.name='CustomNet2'

	def __layers__(self, input_tensor, output_size):
		x_2, x_1, x = self.__head_layer__(input_tensor)

		mid =  [['B2', 128 , 2], ['A', 4, 32], ['Y', 128], 
				['B2', 256 , 1], ['A', 4, 64], ['Y', 256],
				['B2', 256 , 2], ['A', 4, 64], ['Y', 256]]

		x_2, x_1, x = self.__build_mid__(mid, x_2, x_1, x)

		softmax_avg = self.__tail__(x_2, x_1, x, output_size)

		return softmax_avg

if __name__ == '__main__':
	cnn = CustomNet2((32,32),10)
	cnn.summary()