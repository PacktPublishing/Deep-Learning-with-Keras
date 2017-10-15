import matplotlib as mpl

# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')

import pandas as pd
import numpy as np
import os
from keras.layers import Reshape, Flatten, LeakyReLU, Activation, Dense, BatchNormalization, SpatialDropout2D
from keras.layers.convolutional import Conv2D, UpSampling2D, MaxPooling2D, AveragePooling2D
from keras.regularizers import L1L2
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras_adversarial.image_grid_callback import ImageGridCallback

from keras_adversarial import AdversarialModel, simple_gan, gan_targets, fix_names
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
import keras.backend as K
from cifar10_utils import cifar10_data
from image_utils import dim_ordering_unfix, dim_ordering_shape


def model_generator():
    model = Sequential()
    nch = 256
    reg = lambda: L1L2(l1=1e-7, l2=1e-7)
    h = 5
    model.add(Dense(nch * 4 * 4, input_dim=100, kernel_regularizer=reg()))
    model.add(BatchNormalization())
    model.add(Reshape(dim_ordering_shape((nch, 4, 4))))
    model.add(Conv2D(int(nch / 2), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(int(nch / 2), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(int(nch / 4), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(3, (h, h), padding='same', kernel_regularizer=reg()))
    model.add(Activation('sigmoid'))
    return model


def model_discriminator():
    nch = 256
    h = 5
    reg = lambda: L1L2(l1=1e-7, l2=1e-7)

    c1 = Conv2D(int(nch / 4),
                (h, h),
                padding='same',
                kernel_regularizer=reg(),
                input_shape=dim_ordering_shape((3, 32, 32)))
    c2 = Conv2D(int(nch / 2),
                (h, h),
                padding='same',
                kernel_regularizer=reg())
    c3 = Conv2D(nch,
                (h, h),
                padding='same',
                kernel_regularizer=reg())
    c4 = Conv2D(1,
                (h, h),
                padding='same',
                kernel_regularizer=reg())

    def m(dropout):
        model = Sequential()
        model.add(c1)
        model.add(SpatialDropout2D(dropout))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(LeakyReLU(0.2))
        model.add(c2)
        model.add(SpatialDropout2D(dropout))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(LeakyReLU(0.2))
        model.add(c3)
        model.add(SpatialDropout2D(dropout))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(LeakyReLU(0.2))
        model.add(c4)
        model.add(AveragePooling2D(pool_size=(4, 4), padding='valid'))
        model.add(Flatten())
        model.add(Activation('sigmoid'))
        return model
    return m


def example_gan(adversarial_optimizer, path, opt_g, opt_d, nb_epoch, generator, discriminator, latent_dim,
                targets=gan_targets, loss='binary_crossentropy'):
    csvpath = os.path.join(path, "history.csv")
    if os.path.exists(csvpath):
        print("Already exists: {}".format(csvpath))
        return

    print("Training: {}".format(csvpath))
    # gan (x - > yfake, yreal), z is gaussian generated on GPU
    # can also experiment with uniform_latent_sampling
    d_g = discriminator(0)
    d_d = discriminator(0.5)
    generator.summary()
    d_d.summary()
    gan_g = simple_gan(generator, d_g, None)
    gan_d = simple_gan(generator, d_d, None)
    x = gan_g.inputs[1]
    z = normal_latent_sampling((latent_dim,))(x)
    # estiminate z from inputs
    gan_g = Model([x], fix_names(gan_g([z, x]), gan_g.output_names))
    gan_d = Model([x], fix_names(gan_d([z, x]), gan_d.output_names))

    # build adversarial model
    model = AdversarialModel(player_models=[gan_g, gan_d],
                             player_params=[generator.trainable_weights, d_d.trainable_weights],
                             player_names=["generator", "discriminator"])
    model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                              player_optimizers=[opt_g, opt_d],
                              loss=loss)

    # create callback to generate images
    zsamples = np.random.normal(size=(10 * 10, latent_dim))

    def generator_sampler():
        xpred = dim_ordering_unfix(generator.predict(zsamples)).transpose((0, 2, 3, 1))
        print(xpred.shape)
        print(xpred.reshape((10, 10) + xpred.shape[1:]).shape)
        return xpred.reshape((10, 10) + xpred.shape[1:])

    generator_cb = ImageGridCallback(os.path.join(path, "epoch-{:03d}.png"), generator_sampler, cmap=None)

    # train model
    xtrain, xtest = cifar10_data()
    y = targets(xtrain.shape[0])
    ytest = targets(xtest.shape[0])
    callbacks = [generator_cb]
    if K.backend() == "tensorflow":
        callbacks.append(
            TensorBoard(log_dir=os.path.join(path, 'logs'), histogram_freq=0, write_graph=True, write_images=True))
    history = model.fit(x=xtrain, y=y, validation_data=(xtest, ytest),
                        callbacks=callbacks, epochs=nb_epoch,
                        batch_size=32)

    # save history to CSV
    df = pd.DataFrame(history.history)
    df.to_csv(csvpath)

    # save models
    generator.save(os.path.join(path, "generator.h5"))
    d_d.save(os.path.join(path, "discriminator.h5"))


def main():
    # z \in R^100
    latent_dim = 100
    # x \in R^{28x28}
    # generator (z -> x)
    generator = model_generator()
    # discriminator (x -> y)
    discriminator = model_discriminator()
    example_gan(AdversarialOptimizerSimultaneous(), "output/gan-cifar10",
                opt_g=Adam(1e-4, decay=1e-5),
                opt_d=Adam(1e-3, decay=1e-5),
                nb_epoch=1, generator=generator, discriminator=discriminator,
                latent_dim=latent_dim)


if __name__ == "__main__":
    main()
