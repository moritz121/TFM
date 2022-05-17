# Imports
import cv2
import os
import itertools
import numpy as np
from tqdm import tqdm

from preprocess import preprocess
from GAN import GAN

#import pandas as pd

from keras.datasets import mnist

if __name__ == '__main__':

    img_size = (28, 28)
    height, width = img_size

    gan_conf = {'img_size': img_size}

    img_array = preprocess(img_size=img_size)

    X_train = (img_array.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)

    gan = GAN(gan_conf)
    gan.train(X_train)
    '''

    img_size = (28, 28)
    height, width = img_size

    gan_conf = {'img_size': img_size}

    img_array = preprocess(img_size=img_size)

    print(img_array.shape)
    print(img_array.reshape(img_array.shape[0], height*width).shape)

    (X_train, _), (_, _) = mnist.load_data()
    print(X_train[0].shape)
    print(type(X_train[0]))
    print('-----------------------------')
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)
    print(X_train[0].shape)
    print(type(X_train[0]))
    print('-----------------------------')
    '''