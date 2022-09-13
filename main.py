# Imports
import cv2
import os
import itertools
import numpy as np
from tqdm import tqdm

import GAN_3
from GAN import GAN as GAN
from GAN_2 import WGAN as GAN2
from GAN_3 import *
from GAN_4 import GAN_2 as GAN4
from preprocess import preprocess

import torch

def scale_images(images):
	images = images.astype('float32')
	images = (images - 127.5) / 127.5
	return images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    img = cv2.imread('resources/keras_gan/GAN2_9.png')

    img_size_gan_1 = (28, 28, 1)
    img_size_gan_2 = (28, 28, 1)
    img_size_gan_3 = (128, 128, 1)
    img_size_gan_4 = (64, 64, 1)
    latent_dim = 128

    img_array = preprocess(img_size=img_size_gan_4)

    X_train = scale_images(img_array)

    #gan1 = GAN()
    #gan1.train(X_train)
    #exit(0)

    #gan2 = GAN2()
    #gan2.train(500, X_train)
    #exit(0)

    #disc = GAN_3.define_critic(X_train)
    #gen = GAN_3.define_generator(X_train)
    #gan3 = GAN_3.define_GAN(disc, gen)
    #GAN_3.train_wgan(gen, disc, gan3, X_train)
    #exit(0)

    gan_conf = {'data_shape': img_array.shape,
                'latent_dim': latent_dim,
                'n_layers_gen': 1,
                'n_layers_disc': 2,
                'n_epochs': 6000,
                'n_batch': 64}

    gan = GAN4(gan_conf)
    gan.train(X_train)


