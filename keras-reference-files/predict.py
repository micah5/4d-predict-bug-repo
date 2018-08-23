import matplotlib
matplotlib.use('Agg')

import time
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Reshape
from keras.layers import Flatten, BatchNormalization, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Creates the generator model. This model has an input of random noise and
# generates an image that will try mislead the discriminator.
def construct_generator():

    generator = Sequential()

    generator.add(Dense(units=8 * 8 * 512,
                        kernel_initializer='glorot_uniform',
                        input_shape=(1, 1, 100)))
    generator.add(Reshape(target_shape=(8, 8, 512)))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=256, kernel_size=(5, 5),
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=128, kernel_size=(5, 5),
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=64, kernel_size=(5, 5),
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    generator.add(Conv2DTranspose(filters=1, kernel_size=(5, 5),
                                  strides=(2, 2), padding='same',
                                  data_format='channels_last',
                                  kernel_initializer='glorot_uniform'))
    generator.add(Activation('tanh'))

    optimizer = Adam(lr=0.00015, beta_1=0.5)
    generator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=None)

    print('generator')
    generator.summary()

    return generator

def save_generated_figures(generated_images, count):

    for i in range(64):
        image = generated_images[i, :, :, :]
        image += 1
        image *= 127.5
        stacked_img = np.stack((image[:, :, 0],)*3, -1)
        img = stacked_img.astype(np.uint8)
        matplotlib.image.imsave('/output/name%d.png' % (count + i), img)

# Main train function
def predict(image_shape):
    # Build the adversarial model that consists in the generator output
    # connected to the discriminator
    generator = construct_generator()

    if os.path.exists("generator_weights.h5"):
        print('loaded generator model weights')
        generator.load_weights('generator_weights.h5')

    batch_size = 128
    count = 0
    for j in range(100):

        # Generate noise
        noise = np.random.normal(0, 1, size=(batch_size, ) + (1, 1, 100))

        # Generate images
        generated_images = generator.predict(noise)

        for i in range(batch_size):
            image = generated_images[i, :, :, :]
            image += 1
            image *= 127.5
            stacked_img = np.stack((image[:, :, 0],)*3, -1)
            img = stacked_img.astype(np.uint8)
            matplotlib.image.imsave('/output/name%d.png' % (count + i), img)

        #save_generated_figures(generated_images, count)
        count += batch_size + 1

def main():
    image_shape = (128, 128, 1)
    predict(image_shape)

if __name__ == "__main__":
    main()
