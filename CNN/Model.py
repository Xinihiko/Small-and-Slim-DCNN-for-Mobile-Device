from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, BatchNormalization, SeparableConv2D, DepthwiseConv2D, Dropout, Average, MaxPooling2D, Dense, Lambda, add, GlobalAveragePooling2D, Input
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Model:
	def __init__(self, input_size, output_size, init_lr=1e-1, epoch=50, batch=32, optimizers='SGD', load_path=""):
		self.input_size = input_size
		self.output_size = output_size
		self.epoch = epoch
		self.batch = batch
		self.model = self.create_model()
		if load_path != "":
			self.model.load_weights(load_path)
		else:
			if optimizers == 'SGD':
				opt = tf.keras.optimizers.SGD(lr=init_lr, decay=init_lr/epoch)
			elif optimizers == 'ADAM':
				opt = tf.keras.optimizers.Adam(learning_rate=init_lr)
			self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
		self.name=""

	def summary(self):
		self.model.summary()

	def create_model(self):
		(width, height) = self.input_size 
		inputs = Input(shape=[width, height, 3])
		output = self.__layers__(inputs, self.output_size)

		model = tf.keras.Model(inputs=inputs, outputs=output)

		return model

	def __swishLayer__(self, x):
		return Lambda(lambda x: x * tf.math.sigmoid(x))(x)

	def __alphaLayer__(self, inp_x, n, m):
		x = Conv2D(m, kernel_size=1, kernel_initializer='he_uniform')(inp_x)
		x = BatchNormalization()(x)
		x = self.__swishLayer__(x)
		x = DepthwiseConv2D(kernel_size=3, padding='same', depthwise_initializer='he_uniform')(x)
		x = BatchNormalization()(x)
		x = self.__swishLayer__(x)
		x = DepthwiseConv2D(depth_multiplier=n, kernel_size=3, padding='same', depthwise_initializer='he_uniform')(x)
		x = BatchNormalization()(x)
		return self.__swishLayer__(x)

	def __betaLayer__(self, inp_x, n, z):
		x = MaxPooling2D(pool_size=(2,2), strides=2)(inp_x)
		x = Dropout(z)(x)
		x = Conv2D(int(n/4), kernel_size=1, kernel_initializer='he_uniform')(x)
		x = BatchNormalization()(x)
		x = self.__swishLayer__(x)
		x = DepthwiseConv2D(kernel_size=3, padding='same', depthwise_initializer='he_uniform')(x)
		x = BatchNormalization()(x)
		x = self.__swishLayer__(x)
		x = Conv2D(n, kernel_size=1, kernel_initializer='he_uniform')(x)
		x = BatchNormalization()(x)
		x = self.__swishLayer__(x)
		return x

	def __beta2Layer__(self, inp_x, n, z):
		x = Conv2D(int(n/4), strides=z, kernel_size=1, kernel_initializer='he_uniform')(inp_x)
		x = BatchNormalization()(x)
		x = self.__swishLayer__(x)
		x = DepthwiseConv2D(kernel_size=3, padding='same', depthwise_initializer='he_uniform')(x)
		x = BatchNormalization()(x)
		x = self.__swishLayer__(x)
		x = Conv2D(n, kernel_size=1, kernel_initializer='he_uniform')(x)
		x = BatchNormalization()(x)
		x = self.__swishLayer__(x)
		return x

	def __gammaLayer__(self, inp_x, n):
		x = Conv2D(int(n/4), kernel_size=1)(inp_x)
		x = BatchNormalization()(x)
		x = self.__swishLayer__(x)
		x = SeparableConv2D(int(n/4), kernel_size=3, depth_multiplier=2)(x)
		x = BatchNormalization()(x)
		x = self.__swishLayer__(x)
		x = Conv2D(n, kernel_size=1)(inp_x)
		x = BatchNormalization()(x)
		x = self.__swishLayer__(x)
		return x

	def __head_layer__(self, input_x):
		x = Conv2D(8, kernel_size=(5,1), padding='same', kernel_initializer='he_uniform')(input_x)
		x = Conv2D(8, kernel_size=(1,5), padding='same', kernel_initializer='he_uniform')(x)
		x = self.__swishLayer__(x)
		x = Conv2D(16, kernel_size=3, kernel_initializer='he_uniform')(x)
		x = self.__swishLayer__(x)

		x = MaxPooling2D(pool_size=(2,2), strides=2)(x)
		x = Dropout(.2)(x)

		# 2 output
		x_1 = Conv2D(16, kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
		x_1 = Conv2D(32, kernel_size=3, padding='same', kernel_initializer='he_uniform')(x_1)
		x_1 = BatchNormalization()(x_1)
		x_1 = self.__swishLayer__(x_1)

		x = Conv2D(16, kernel_size=1, kernel_initializer='he_uniform')(x)
		x = Conv2D(32, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_uniform')(x)
		x = BatchNormalization()(x)
		x = self.__swishLayer__(x)

		# 3 output
		x_2 = Conv2D(32, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_uniform')(x_1)
		x_2 = Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_uniform')(x_2)
		x_2 = BatchNormalization()(x_2)
		x_2 = self.__swishLayer__(x_2)

		x_1 = Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_uniform')(x_1)
		x_s = Conv2D(64, kernel_size=1, kernel_initializer='he_uniform')(x)
		x_1 = add([x_1,x_s])
		x_1 = BatchNormalization()(x_1)
		x_1 = self.__swishLayer__(x_1)

		x = Conv2D(32, kernel_size=1, kernel_initializer='he_uniform')(x)
		x = Conv2D(64, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_uniform')(x)
		x = BatchNormalization()(x)
		x = self.__swishLayer__(x)

		return x_2, x_1, x

	def __build_mid__(self, mid, x_2, x_1, x):
		for i in range(len(mid)):
			if mid[i][0] == 'B':
				x_2 = self.__betaLayer__(x_2, mid[i][1], mid[i][2])
				x_1 = self.__betaLayer__(x_1, mid[i][1], mid[i][2])
				x = self.__betaLayer__(x, mid[i][1], mid[i][2])
			elif mid[i][0] == 'B2':
				x_2 = self.__beta2Layer__(x_2, mid[i][1], mid[i][2])
				x_1 = self.__beta2Layer__(x_1, mid[i][1], mid[i][2])
				x = self.__beta2Layer__(x, mid[i][1], mid[i][2])
			elif mid[i][0] == 'A':
				x_2 = self.__alphaLayer__(x_2, mid[i][1], mid[i][2])
				x_1 = self.__alphaLayer__(x_1, mid[i][1], mid[i][2])
				x = self.__alphaLayer__(x, mid[i][1], mid[i][2])
			elif mid[i][0] == 'Y':
				x_a = Lambda(lambda x: 1 * x)(x_2)
				x_b = Lambda(lambda x: 1 * x)(x_1)
				x_c = Lambda(lambda x: 1 * x)(x)

				x_2 = self.__gammaLayer__(x_2, mid[i][1])
				x_1 = self.__gammaLayer__(x_1, mid[i][1])
				x = self.__gammaLayer__(x, mid[i][1])

				x_2 = add([x_a,x_2])
				x_1 = add([x_b,x_1])
				x = add([x_c,x])

		return x_2, x_1, x

	def __tail__(self, x_2, x_1, x, output_size, dropout=False):
		x_2 = GlobalAveragePooling2D()(x_2)
		x_1 = GlobalAveragePooling2D()(x_1)
		x = GlobalAveragePooling2D()(x)

		if dropout:
			x_2 = Dropout(.5)(x_2)
			x_1 = Dropout(.5)(x_1)
			x = Dropout(.5)(x)

		x_2 = Dense(output_size, activation='softmax', name='out_2')(x_2)
		x_1 = Dense(output_size, activation='softmax', name='out_1')(x_1)
		x = Dense(output_size, activation='softmax', name='out')(x)

		softmax_avg = Average()([x_2,x_1,x])
		return softmax_avg

	def train(self, trainImg, trainLbl, testImg, testLbl, aug=True, plot=True, classWeight=None):
		eS = EarlyStopping(monitor='val_loss', patience=int(.3*self.epoch), verbose=0, mode='min')
		mChk = ModelCheckpoint(self.name+'.h5', save_best_only=True, monitor='val_loss', mode='min')
		rLR = ReduceLROnPlateau(monitor='val_loss', factor=.01, patience=int(.3*self.epoch), verbose=1, min_lr=1e-6, mode='min')

		h = ""

		if aug:
			aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, fill_mode='nearest', horizontal_flip=True)
			h = self.model.fit(aug.flow(trainImg, trainLbl, batch_size=self.batch),
							validation_data=(testImg, testLbl),
							steps_per_epoch=len(trainImg)//self.batch,
							epochs=self.epoch, callbacks=[eS, mChk, rLR],
							class_weight=classWeight, verbose=2)
		else:
			h = self.model.fit(	trainImg, trainLbl, 
							batch_size=self.batch,
							validation_data=(testImg, testLbl),
							steps_per_epoch=len(trainImg)//self.batch,
							epochs=self.epoch, callbacks=[eS, mChk, rLR],
							class_weight=classWeight, verbose=2)

		if plot:
			self.plotHistory(h)
		pass

	def plotHistory(self, h):
		self.plotLoss(h)
		self.plotAcc(h)

	def plotLoss(self, h):
		# print(h.history)
		n = np.array(range(len(h.history['loss'])))
		plt.style.use('ggplot')
		plt.figure()
		plt.plot(n, h.history['loss'], label='Train_loss')
		plt.plot(n, h.history['val_loss'], label='Val_loss')
		plt.title=('Training Accuracy and Loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss/Accuracy')
		plt.legend(loc='lower left')
		plt.savefig('Result Loss '+self.name+'.png')

	def plotAcc(self, h):
		# print(h.history)
		n = np.array(range(len(h.history['loss'])))
		plt.style.use('ggplot')
		plt.figure()
		plt.plot(n, h.history['accuracy'], label='Train_acc')
		plt.plot(n, h.history['val_accuracy'], label='Val_acc')
		plt.title=('Training Accuracy and Loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss/Accuracy')
		plt.legend(loc='lower left')
		plt.savefig('Result Acc '+self.name+'.png')

	def test(self, img, lbl, classNames=None, f1=False, savefig=True, save_path=""):
		if '/' not in save_path and '\\' not in save_path:
			save_path += '/'
		prediction = self.model.predict(img, batch_size=self.batch)
		print(lbl.argmax(axis=1), prediction.argmax(axis=1))
		if f1:
			print(classification_report(lbl.argmax(axis=1), prediction.argmax(axis=1), target_names = classNames))
		if savefig:
			cm = confusion_matrix(lbl.argmax(axis=1), prediction.argmax(axis=1))
			cm_display = ConfusionMatrixDisplay(cm).plot()
			cm_display.figure_.savefig(save_path+'Confusion Matrix {}.png'.format(self.name))
		