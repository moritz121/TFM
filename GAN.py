import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import keras
from keras.layers import Dense, Dropout, Input, Reshape, Flatten
from keras.models import Model,Sequential
from keras.datasets import mnist
from tqdm import tqdm
#from keras.layers.advanced_activations import LeakyReLU
from keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam

from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.convolutional import UpSampling2D, Conv2D

class GAN():
    def __init__(self):
        self.img_size = (28,28, 1)
        self.channels = 1
        self.latent_dim = 100
    '''
    def discriminator(self, loss, nodes):
        discriminator = Sequential()

        discriminator.add(Dense(units=nodes[0], input_dim=self.img_size[0]*self.img_size[1]))
        discriminator.add(LeakyReLU(0.2))

        discriminator.add(Dense(units=nodes[1]))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Dense(units=nodes[2]))
        discriminator.add(LeakyReLU(0.2))

        discriminator.add(Dense(units=1, activation='sigmoid'))

        discriminator.compile(loss=loss, optimizer=self.optimizer_adam())

        return discriminator

    def generator(self, loss, nodes):
        generator = Sequential()

        generator.add(Dense(units=nodes[2], input_dim=100))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(units=nodes[1]))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(units=nodes[0]))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(units=self.img_size[0]*self.img_size[1], activation='tanh'))
        generator.compile(loss=loss, optimizer=self.optimizer_adam())
        return generator

    '''

    def discriminator(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_size, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_size)
        validity = model(img)

        return Model(img, validity)

    def generator(self, loss, nodes):
        model = Sequential()

        model.add(Dense(7 * 7 * 128, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))
        #model.add(Reshape((28, 28)))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def gan(self, generator, discriminator, loss):
        discriminator.trainable = False
        gen_input = Input(shape=(100,))
        fake = generator(gen_input)
        gan_output = discriminator(fake)
        gan = Model(inputs=gen_input, outputs=gan_output)
        gan.compile(loss=loss, optimizer=self.optimizer_adam(), metrics=['accuracy'])
        return gan

    def train(self, X_train, epochs=10, batch_size=128):

        y_gen = np.ones((batch_size, 1))
        noise = np.random.normal(0, 1, [batch_size, self.latent_dim])

        nodes = [256, 512, 1024]

        loss = self.loss_binary_cross()

        generator = self.generator(loss, nodes=nodes)
        generator.summary()

        discriminator = self.discriminator(loss, nodes=nodes)
        discriminator.summary()
        discriminator.compile(loss=loss, optimizer=self.optimizer_adam(), metrics=['accuracy'])

        gan = self.gan(generator=generator, discriminator=discriminator, loss=loss)
        gan.summary()

        batch_count = X_train.shape[0] / batch_size

        for e in range(1, epochs+1):
            for i in range(batch_size):

                noise = np.random.normal(0, 1, [batch_size, 100])

                # Gen img from noise input

                gen_img = generator.predict(noise)

                # Get random set of real img

                image_batch = X_train[np.random.randint(low=0, high=X_train.shape[0], size=batch_size)]

                # Mix fake and real

                X = np.concatenate([image_batch, gen_img])

                y_dis = np.zeros(2*batch_size)
                y_dis[:batch_size] = 0.9

                # Pre-train discriminator
                discriminator.trainable=True
                discriminator.train_on_batch(X, y_dis)

                noise= np.random.normal(0,1, [batch_size, 100])
                y_gen = np.ones((batch_size, 1))

                # Fix discriminator weights
                discriminator.trainable = False

                # Train GAN
                history = gan.train_on_batch(noise, y_gen)

                #print(gan.metrics_names)
                print('Epoch ' + str(e) + ' Batch ' + str(i) + ': Loss -> ' + str(history[0]) + ' | Accuracy -> ' + str(history[1]))

                if e == 1 or e% 20 == 0:
                    self.plot_generated_images(e, generator)

    def optimizer_adam(self):
        return Adam(learning_rate=0.0002, beta_1=0.5)

    def loss_binary_cross(self):
        return 'binary_crossentropy'

    def plot_generated_images(self, epoch, generator, examples=100, dim=(10,10), figsize=(10,10)):

        noise = np.random.normal(loc=0, scale=1, size=[examples, 100])
        generated_images = generator.predict(noise)
        generated_images = generated_images.reshape(100, 28, 28)

        cv2.imwrite('resources/gan/gan_generated_image %d.png' % epoch, generated_images[-1])

        """
        plt.figure(figsize=figsize)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i + 1)
            plt.imshow(generated_images[i], interpolation='nearest')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('resources/gan/gan_generated_image %d.png' % epoch)
        plt.close('all')
        """

