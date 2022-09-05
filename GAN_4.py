from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout

#from tensorflow.keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.keras.layers import Conv2D

import matplotlib.pyplot as plt

from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K

import cv2
import numpy as np
from art import tprint
import tensorflow as tf

import gc

class GAN_2():

    def __init__(self, conf):
        self.img_shape = conf['data_shape'][1:4]
        self.dataset_size = conf['data_shape'][0]
        self.latent_dim = conf['latent_dim']
        self.n_layers_disc = conf['n_layers_disc']
        self.n_layers_gen = conf['n_layers_gen']
        self.n_epochs = conf['n_epochs']
        self.n_batch = conf['n_batch']
        self.channels = 1

    def build_discriminator(self):

        model = Sequential()

        for x in range(self.n_layers_disc):
            # Layer 1 disc
            model.add(Conv2D(64, (3, 3), padding='same', input_shape=self.img_shape))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
            model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)

        return model


    def build_generator(self):

        model = Sequential()
        n_nodes = 256 * 4 * 4
        model.add(Dense(n_nodes, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((4, 4, 256)))

        for x in range(self.n_layers_gen):
            # Layer 1 gen
            model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
            model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(1, (3,3), activation='tanh', padding='same'))

        return model


    def build_gan(self, disc, gen):

        opt = Adam(learning_rate=0.0002, beta_1=0.5)

        disc.summary()
        gen.summary()
        disc.trainable = False
        model = Sequential()
        model.add(gen)
        model.add(disc)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    def train(self, X_train):

        gen = self.build_generator()
        disc = self.build_discriminator()

        gan_model = self.build_gan(disc, gen)
        tprint('GENERATOR')
        gen.summary()
        tprint('DISCRIMINATOR')
        disc.summary()
        tprint('TRAINING')

        bat_per_epo = int(X_train.shape[0] / self.n_batch)
        half_batch = int(self.n_batch / 2)
        for i in range(self.n_epochs):
            for j in range(bat_per_epo):

                X_real, y_real = self.generate_real_samples(X_train, half_batch)
                d_loss1 = disc.train_on_batch(X_real, y_real)
                X_fake, y_fake = self.generate_fake_samples(gen, half_batch)
                d_loss2 = disc.train_on_batch(X_fake, y_fake)
                X_gan = self.generate_latent_points(self.n_batch)
                y_gan = ones((self.n_batch, 1))
                g_loss = gan_model.train_on_batch(X_gan, y_gan)
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                      (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))

            if (i + 1) % 10 == 0:
                self.summarize_performance(i, gen, disc, X_train, self.latent_dim)

    def generate_real_samples(self, dataset, n_samples):
        ix = randint(0, dataset.shape[0], n_samples)
        X = dataset[ix]
        y = ones((n_samples, 1))
        return X, y

    def generate_latent_points(self, n_samples):
        x_input = randn(self.latent_dim * n_samples)
        x_input = x_input.reshape(n_samples, self.latent_dim)
        return x_input

    def generate_fake_samples(self, g_model, n_samples):
        x_input = self.generate_latent_points(n_samples)
        X = g_model.predict(x_input)
        y = zeros((n_samples, 1))
        return X, y

    def summarize_performance(self, epoch, g_model, d_model, dataset, n_samples=150):
        X_real, y_real = self.generate_real_samples(dataset, n_samples)
        acc_real = d_model.evaluate(X_real, y_real, verbose=0)
        x_fake, y_fake = self.generate_fake_samples(g_model, n_samples)
        acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
        print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
        self.save_plot(x_fake, epoch)
        filename = 'model/generator_model_%03d.h5' % (epoch + 1)
        d_filename = 'model/discriminator_model_%03d.h5' % (epoch + 1)
        g_model.save(filename)
        d_model.save(d_filename)

    def save_plot(self, x_fake, epoch):

        r, c = 5, 5

        fig, axs = plt.subplots(r, c)
        cnt = 0

        for i in range(r):
            for j in range(c):
                img = cv2.normalize(x_fake[cnt], None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axs[i,j].imshow(img)
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("resources/keras_gan/GAN_%d.png" % epoch)
        plt.close(fig)
