from Datasets.GTSRB import GTSRB
from Datasets.CIFAR10 import CIFAR10
from Datasets.CIFAR100 import CIFAR100
# from Datasets.ImageNet import ImageNet
from Datasets.SVHN import SVHN

from CNN.CustomNet import CustomNet
from CNN.CustomNet2 import CustomNet2

datasets = ["GTSRB", "CIFAR10", "CIFAR100", "SVHN"]
for dt in datasets:
	dataset = None
	if dt == 'GTSRB':
		dataset = GTSRB()
	if dt == 'SVHN':
		dataset = SVHN()
	if dt == 'CIFAR10':
		dataset = CIFAR10()
	if dt == 'CIFAR100':
		dataset = CIFAR100()

	input_size = dataset.trainX.shape[1:3]
	output_size = dataset.trainY.shape[-1]

	best_model = 'New Result/'
	md = ['CustomNet', 'CustomNet2']

	model = CustomNet(input_size, output_size, epoch=150, optimizers='ADAM', batch=64, init_lr=0.001, load_path=best_model+dt+'/'+md[0]+'.h5')
	model.test(dataset.testX, dataset.testY, classNames=dataset.classLabels, f1=True, savefig=True, save_path=best_model+dt+'/')

	model = CustomNet2(input_size, output_size, epoch=150, optimizers='ADAM', batch=64, init_lr=0.001, load_path=best_model+dt+'/'+md[1]+'.h5')
	model.test(dataset.testX, dataset.testY, classNames=dataset.classLabels, f1=True, savefig=True, save_path=best_model+dt+'/')


# model = CustomNet(input_size, output_size, epoch=150, optimizers='ADAM', batch=64, init_lr=0.001)
# model.train(dataset.trainX, Datasetset.trainY, dataset.testX, dataset.testY)

# model = CustomNet2(input_size, output_size, epoch=150, optimizers='ADAM', batch=64, init_lr=0.001)
# model.train(dataset.trainX, dataset.trainY, dataset.testX, dataset.testY)

# model = CNN_Extra(dataset.trainX.shape[1:3], dataset.trainY.shape[-1], epoch=250, init_lr=0.01)
# model.train(dataset.trainX, dataset.trainY, dataset.testX, dataset.testY, dataset.classList, classWeight=dataset.classWeight)
# model = CNN_Extra(dataset.trainX.shape[1:3], dataset.trainY.shape[-1], epoch=250, init_lr=0.01, load_path='CNN_Extra.h5')
# model.test(dataset.testX, dataset.testY, classNames=dataset.classList, f1=True, savefig=True)