from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Add, DepthwiseConv2D, GlobalAveragePooling2D, Reshape, \
    Multiply, BatchNormalization, Activation, Layer, LeakyReLU, Dropout, UpSampling2D, Conv2DTranspose, Flatten, Dense
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, RMSprop
from time import time
from tensorflow.keras.initializers import RandomNormal
from utilities import *
import matplotlib as pyplot
import numpy as np

def define_critic(input, depth=64):
    # Initialisation
    model = Sequential(name="Critic")
    init = RandomNormal(stddev=0.02)
    const = ClipConstraint(0.01)
    shape = input.shape
    shape = (shape[1], shape[2], shape[3])

    # Normal convolution
    model.add(
        Conv2D(depth, (3, 3), input_shape=shape, padding='same', kernel_initializer=init, kernel_constraint=const))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # Downsample to 32x32
    model.add(
        Conv2D(depth * 2, (3, 3), strides=(4, 4), padding='same', kernel_initializer=init, kernel_constraint=const))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # Downsample to 4x4
    model.add(
        Conv2D(depth * 4, (3, 3), strides=(4, 4), padding='same', kernel_initializer=init, kernel_constraint=const))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # Output
    model.add(Flatten())
    model.add(Dense(1))
    optimiser = RMSprop(lr=0.0005)
    model.compile(loss=wasserStein_loss, optimizer=optimiser)
    return model

def define_generator(input, depth=256, dim=4):
    model = Sequential(name="Generator")
    init = RandomNormal(stddev=0.02)
    nodes = 32 * dim ** 2
    model.add(Dense(nodes, input_dim=100, kernel_initializer=init))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((dim, dim, 32)))
    # upsample to 8x8
    model.add(UpSampling2D())
    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(UpSampling2D())
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(UpSampling2D())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 64x64
    model.add(UpSampling2D())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # Upsample to 128x128
    model.add(UpSampling2D())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    # final output to 128x128x3
    model.add(Conv2D(1, (3, 3), padding='same', activation='tanh', kernel_initializer=init))
    return model

def define_GAN(critic, generator):
    model = Sequential()
    critic.trainable = False
    model.add(generator)
    model.add(critic)
    optimiser = RMSprop(lr=0.0005)
    model.compile(loss=wasserStein_loss, optimizer=optimiser)
    return model

def train_wgan(generator, critic, wgan, data, latent_dim=100, n_epochs=300, batch_size=64, critic_iterations=5):
    # batches per epoch
    batch_epoch = int(data.shape[0] / batch_size)
    n_steps = batch_epoch * n_epochs
    half = int(batch_size / 2)
    print('batch per epoch %d steps %d batch half %d' % (batch_epoch, n_steps, half))

    loss_1, loss_2, loss_gan = list(), list(), list()
    for i in range(n_steps):
        loss1_temp, loss2_temp = list(), list()
        for _ in range(critic_iterations):
            X_real, Y_real = generate_real_samples(data, half)
            loss1 = critic.train_on_batch(X_real, Y_real)
            loss1_temp.append(loss1)
            X_fake, Y_fake = generate_samples(generator, half)
            loss2 = critic.train_on_batch(X_fake, Y_fake)
            loss2_temp.append(loss2)
        loss_1.append(np.mean(loss1_temp))
        loss_2.append(np.mean(loss2_temp))
        X_gan = generate_latent_points(batch_size)
        Y_gan = -np.ones((batch_size, 1))
        loss_g = wgan.train_on_batch(X_gan, Y_gan)
        loss_gan.append(loss_g)

        # Save loss and accuracy on this batch
        # print('step %d c1=%.3f c2=%.3f g=%.3f' % (i+1, loss_1[-1], loss_2[-1], loss_g))
        if (i + 1) % batch_epoch == 0:
            plot_history(loss_1, loss_2, loss_gan, True)
            summarise_performance((i + 1) / batch_epoch, generator)
    plot_history(loss_1, loss_2, loss_gan, True)

def summarize_performance(step, g_model, latent_dim, n_samples=100):
	# prepare fake examples
	X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# scale from [-1,1] to [0,1]
	X = (X + 1) / 2.0
	# plot images
	for i in range(10 * 10):
		# define subplot
		pyplot.subplot(10, 10, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
	# save plot to file
	filename1 = 'generated_plot_%04d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%04d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels with 1.0 for 'fake'
	y = np.ones((n_samples, 1))
	return X, y

def plot_history(d1_hist, d2_hist, g_hist):
	# plot history
	pyplot.plot(d1_hist, label='crit_real')
	pyplot.plot(d2_hist, label='crit_fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	pyplot.savefig('plot_line_plot_loss.png')
	pyplot.close()