"""
The :mod:`unet` module generates and saves a neural network
with unet architecture"""
# Author: Zizhao Zhang, Ingo Guehring

############################random#####################
# Seed value
# Apparently you may use different seed values at each stage
seed_value = 0

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";

# The GPU id to use, in our setup it is in [0, 1, 2]
# 0,1 = GP 100, 2 = GV 100 (new)
# this needs to be adapted by hand
os.environ["CUDA_VISIBLE_DEVICES"] = "0";

os.environ['PYTHONHASHSEED'] = str(seed_value)
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from tensorflow.keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import math

####################end random ####################################

# imports for model
from tensorflow.keras import models
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D,
    Cropping2D, concatenate, ZeroPadding2D,
    Lambda, LeakyReLU
)
from tensorflow.keras.optimizers import Adam

# imports for training
import pickle as pickle
import argparse
import numpy as np
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau,
    CSVLogger
)

from tensorflow.keras.models import model_from_json, load_model


from models.losses import (
    l2_on_wedge_factory, l1_l2, l1_weighted_sum_factory,
    l1_TV_factory, my_mean_squared_error, my_psnr, CUSTOM_OBJECTS
)
#from models.callbacks import PlotLearning


class UNet():
    """Build Unet neural network architecture

    The easiest way to use a unet for the inpainting task is to use masked
    sinograms as input (i.e. the region where the inpainting should be done
    is masked) and train the network to produce the complete sinogram.

    There are several choices for a loss function in this setup:
        (1) least square error on the whole sinogram
        (2) least square only on the inpainted region
        (3) mix (1) and (2) and give different weightings for the l2 error
            in different regions
    """
    def __init__(self):
        print('build UNet ...')

    def get_crop_shape(self, target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

    def create_model(self, img_shape, activation='relu',
                     kernel_initializer='glorot_normal',
                     learning_rate=None,
                     loss='mse',
                     pretrained_weights=False):
        """Build a unet

        Parameters
        -----------
        img_shape : tuple of int, (x_dim, y_dim, n_channels).
            In our case x_dim is the number of angles, y_dim the number of
            translations, and n_channels=1.

        pretrained_weights: str, path to weights that should be loaded
            (not tested yet)

        Returns:
        -----------

        model: unet
        """
        concat_axis = 3
        inputs = Input(shape=img_shape)


        conv1 = Conv2D(32, (3, 3), activation=activation, padding='same',
                       kernel_initializer=kernel_initializer,
                       name='conv1_1')(inputs)

        conv1 = Conv2D(32, (3, 3), activation=activation,
                       kernel_initializer=kernel_initializer,
                       padding='same')(conv1)

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation=activation,
                       kernel_initializer=kernel_initializer,
                       padding='same')(pool1)

        conv2 = Conv2D(64, (3, 3), activation=activation,
                       kernel_initializer=kernel_initializer,
                       padding='same')(conv2)

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation=activation,
                       kernel_initializer=kernel_initializer,
                       padding='same')(pool2)

        conv3 = Conv2D(128, (3, 3), activation=activation,
                       kernel_initializer=kernel_initializer,
                       padding='same')(conv3)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation=activation,
                       kernel_initializer=kernel_initializer,
                       padding='same')(pool3)

        conv4 = Conv2D(256, (3, 3), activation=activation,
                       kernel_initializer=kernel_initializer,
                       padding='same')(conv4)

        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation=activation,
                       kernel_initializer=kernel_initializer,
                       padding='same')(pool4)

        conv5 = Conv2D(512, (3, 3), activation=activation,
                       kernel_initializer=kernel_initializer,
                       padding='same')(conv5)

        up_conv5 = UpSampling2D(size=(2, 2))(conv5)
        ch, cw = self.get_crop_shape(conv4, up_conv5)
        crop_conv4 = Cropping2D(cropping=(ch, cw))(conv4)
        up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
        conv6 = Conv2D(256, (3, 3), activation=activation,
                       kernel_initializer=kernel_initializer,
                       padding='same')(up6)
        # conv6 = LeakyReLU(alpha=0.2)(conv6)

        conv6 = Conv2D(256, (3, 3), activation=activation,
                       kernel_initializer=kernel_initializer,
                       padding='same')(conv6)
        # conv6 = LeakyReLU(alpha=0.2)(conv6)

        up_conv6 = UpSampling2D(size=(2, 2))(conv6)
        ch, cw = self.get_crop_shape(conv3, up_conv6)
        crop_conv3 = Cropping2D(cropping=(ch, cw))(conv3)
        up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
        conv7 = Conv2D(128, (3, 3), activation=activation,
                       kernel_initializer=kernel_initializer,
                       padding='same')(up7)
        # conv7 = LeakyReLU(alpha=0.2)(conv7)

        conv7 = Conv2D(128, (3, 3), activation=activation,
                       kernel_initializer=kernel_initializer,
                       padding='same')(conv7)
        # conv7 = LeakyReLU(alpha=0.2)(conv7)

        up_conv7 = UpSampling2D(size=(2, 2))(conv7)
        ch, cw = self.get_crop_shape(conv2, up_conv7)
        crop_conv2 = Cropping2D(cropping=(ch, cw))(conv2)
        up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
        conv8 = Conv2D(64, (3, 3), activation=activation,
                       kernel_initializer=kernel_initializer,
                       padding='same')(up8)
        # conv8 = LeakyReLU(alpha=0.2)(conv8)

        conv8 = Conv2D(64, (3, 3), activation=activation,
                       kernel_initializer=kernel_initializer,
                       padding='same')(conv8)
        # conv8 = LeakyReLU(alpha=0.2)(conv8)

        up_conv8 = UpSampling2D(size=(2, 2))(conv8)
        ch, cw = self.get_crop_shape(conv1, up_conv8)
        crop_conv1 = Cropping2D(cropping=(ch, cw))(conv1)

        up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
        conv9 = Conv2D(32, (3, 3), activation=activation,
                       kernel_initializer=kernel_initializer,
                       padding='same')(up9)
        # conv9 = LeakyReLU(alpha=0.2)(conv9)

        conv9 = Conv2D(32, (3, 3), activation=activation,
                       kernel_initializer=kernel_initializer,
                       padding='same')(conv9)
        # conv9 = LeakyReLU(alpha=0.2)(conv9)

        ch, cw = self.get_crop_shape(inputs, conv9)

        # If the sinogram size is not such that the consecutive convs and
        # max-pooling yields even values of x and y
        # (i.e. width and height of the feature map) at each stage,
        # then this zero padding here is needed. It is responsible for
        # introducing (almost) zero predictions at the border of the sinograms
        conv9 = ZeroPadding2D(padding=((ch[0], ch[1]), (cw[0], cw[1])))(conv9)
        conv10 = Conv2D(1, (1, 1))(conv9)
        
        model = models.Model(inputs=inputs, outputs=conv10)

        metrics = [my_mean_squared_error, 'mse', 'mae',
                   l2_on_wedge_factory(None), my_psnr]
        optimizer = Adam(lr=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        if(pretrained_weights):
            model.load_weights(pretrained_weights)
        return model

