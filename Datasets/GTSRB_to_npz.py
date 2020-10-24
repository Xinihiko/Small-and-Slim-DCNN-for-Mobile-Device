# part of this code is taken from 
# https://github.com/chsasank/Traffic-Sign-Classification.keras/blob/master/Traffic%20Sign%20Classification.ipynb
# modified to fit the task, which is convert the dataset to npz for faster loading.

import numpy as np
from skimage import io, color, exposure, transform
import os
import glob
import pandas as pd

NUM_CLASSES = 43
IMG_SIZE = 32

def preprocess_img(img):
    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    return img

def get_class(img_path):
    img_path = img_path.replace('\\','/')
    return int(img_path.split('/')[-2])

def load_train():
    root_dir = 'GTSRB/Final_Training/Images/'
    imgs = []
    labels = []

    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
    np.random.shuffle(all_img_paths)
    for img_path in all_img_paths:
        try:
            img = preprocess_img(io.imread(img_path))
            label = get_class(img_path)
            imgs.append(img)
            labels.append(label)

            if len(imgs)%1000 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
        except (IOError, OSError):
            print('missed', img_path)
            pass

    X = np.array(imgs, dtype='float32')
    Y = np.eye(NUM_CLASSES, dtype='uint8')[labels]

    np.savez_compressed('GTSRB-train.npz', images=X, labels=Y)

def load_test():
    test = pd.read_csv('GTSRB/GT-final_test.csv',sep=';')

    X_test = []
    y_test = []
    i = 0
    for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join('GTSRB/Final_Test/Images/',file_name)
        X_test.append(preprocess_img(io.imread(img_path)))
        y_test.append(class_id)
        
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y = np.eye(NUM_CLASSES, dtype='uint8')[y_test]

    np.savez_compressed('GTSRB-test.npz', images=X_test, labels=y)

if __name__ == '__main__':
    # load_train()
    load_test()