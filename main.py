# Imports
import cv2
import os
import itertools
import numpy as np
from tqdm import tqdm

from misc import split_test_train
from GAN_2 import GAN_2 as GAN2
from preprocess import preprocess

def scale_images(images):
	# convert from unit8 to float32
	images = images.astype('float32')
	# scale from [0,255] to [-1,1]
	images = (images - 127.5) / 127.5
	return images

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    img_size = (64, 64)
    latent_dim = 128

    img_array = preprocess(img_size=img_size)

    X_train = scale_images(img_array)

    gan_conf = {'data_shape': img_array.shape,
                'latent_dim': latent_dim,
                'n_layers_gen': 1,
                'n_layers_disc': 2,
                'n_epochs': 1500,
                'n_batch': 64}

    gan = GAN2(gan_conf)
    gan.train(X_train)



