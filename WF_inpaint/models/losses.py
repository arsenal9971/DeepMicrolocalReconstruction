"""
The :mod:`losses` contains customized loss functions for training of an
inpainting learning algorithm on sinograms"""
# Author: Ingo Guehring
import numpy as np

import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow.keras.backend import mean
from tensorflow.image import psnr

def l1_sum_wedge_nonwedge(y_true, y_pred, wedge_mask):
    """factor * l1_wedge + l2_nonwedge

    The resulting loss function thus depends on the region to be inpainted.
    Since in tensorflow a loss function has only two input (y_true, y_pred)
    we need two function to define our parametrized loss function

    The weighting factor has to be set by hand is not saved in training!

    Parameters
    -----------
    y_true : numpy array of size nsamples x angels x translations x 1,
        the complete sinogram

    y_pred : numpy array of size nsamples x angels x translations x 1,
        the inpainted sinogram

    wedge_mask:

    Returns:
    -----------

    factor * l1 difference of inpainted region + ...
    """
    return (8*losses.mean_absolute_error(
        y_pred * (1 - wedge_mask[tf.newaxis, :, tf.newaxis, tf.newaxis]),
        y_true * (1 - wedge_mask[tf.newaxis, :, tf.newaxis, tf.newaxis])) +
      losses.mean_absolute_error(
              y_pred * wedge_mask[tf.newaxis, :, tf.newaxis, tf.newaxis],
              y_true * wedge_mask[tf.newaxis, :, tf.newaxis, tf.newaxis]))


def l2_on_wedge_mask(y_true, y_pred, wedge_mask):
    """Loss functions depending on the missing wedge. Only the region
    to be inpainted is used for the l2 difference.

    The next two functions define a loss for the trainign of the unet,
    where only the inpainted region is taken into account for the l2
    difference.
    The resulting loss function thus depends on the region to be inpainted.
    Since in tensorflow a loss function has only two input (y_true, y_pred)
    we need two function to define our parametrized loss function

    Parameters
    -----------
    y_true : numpy array of size nsamples x angels x translations x 1,
        the complete sinogram

    y_pred : numpy array of size nsamples x angels x translations x 1,
        the inpainted sinogram

    wedge_mask:

    Returns:
    -----------

    l2 difference of inpainted region
    """
    # Ok, no premature optimization this now hardcoded and only valid for
    # one setting!!

    #print(y_true.shape)
    #print(wedge_mask.shape)
    #samples = np.full(y_true.shape[0], True, dtype=bool)
    #translations = np.full(y_true.shape[2], True, dtype=bool)
    #channel = np.array([True])
    #mask = np.outer(samples, translations)

    #print(y_true[:, ~wedge_mask, :, :].shape)
    #ind_mask = np.arange(len(wedge_mask))[~wedge_mask]
    # return np.mean((y_true[:, ~wedge_mask, :, :] - y_pred[:, ~wedge_mask, :, :])**2)

    # return tf.reduce_sum(losses.mean_squared_error(y_true, y_pred)) +
    # tf.reduce_sum(losses.mean_squared_error(y_true[:, 34:54, :, :],
    # y_pred[:, 34:54, :, :]))
    return losses.mean_squared_error(y_true[:, 34:54, :, :],
                                     y_pred[:, 34:54, :, :])


def l2_on_wedge_mask_numpy(y_true, y_pred, wedge_mask):
    """Loss functions depending on the missing wedge. Only the region
    to be inpainted is used for the l2 difference.

    This function cannot be used in tensorflow because of the slicing!!!

    Parameters
    -----------
    y_true : numpy array of size nsamples x angels x translations x 1,
        the complete sinogram

    y_pred : numpy array of size nsamples x angels x translations x 1,
        the inpainted sinogram

    wedge_mask:

    Returns:
    -----------

    l2 difference of inpainted region
    """
    return losses.mean_squared_error(y_true[:, ~wedge_mask, :],
                                     y_pred[:, ~wedge_mask, :])


def l1_TV_factory(TV_weight):
    """Takes a parameter and returns a loss function compliant with tensorflow
    guidelines"""
    def l1_TV(y_true, y_pred):
        return losses.mean_absolute_error(y_true, y_pred) + \
            TV_weight * tf.reduce_sum(tf.image.total_variation(y_pred))
    return l1_TV


def l2_on_wedge_factory(wedge_mask):
    """Takes a parameter and returns a loss function compliant with tensorflow
    guidelines"""
    def l2_on_wedge(y_true, y_pred):
        return l2_on_wedge_mask(y_true, y_pred, wedge_mask)
    return l2_on_wedge


def l1_weighted_sum_factory(wedge_mask):
    """Takes a parameter and returns a loss function compliant with tensorflow
    guidelines"""
    def l1_weighted_sum(y_true, y_pred):
        return l1_sum_wedge_nonwedge(y_true, y_pred, wedge_mask)
    return l1_weighted_sum


def wasserstein(y_true, y_pred):
    return mean(y_true * y_pred)


def l1_l2(y_true, y_pred):
    return losses.mean_squared_error(y_true, y_pred) + \
            losses.mean_absolute_error(y_true, y_pred)


def my_mean_squared_error(y_true, y_pred):
    # return K.mean(K.square(y_pred - y_true), axis=[1, 2])
    return K.mean(K.square(y_pred - y_true))


def max_psnr_sino(x_size, y_size, path_to_data, impl):
    ray_trafo_full, _, _ = get_trafos_and_wedge_mask_from_key(
        path_to_data=path_to_data, impl=impl)
    return np.max(ray_trafo_full(np.ones(x_size, y_size)))


def my_psnr(a, b):
    return psnr(a, b, max_val=5.2)


def l1_and_l2_loss(y_true,y_pred):
    a = losses.mean_squared_error(y_true,y_pred)
    b = losses.mean_absolute_error(y_true,y_pred)
    return a

CUSTOM_OBJECTS = {
    'my_mean_squared_error': my_mean_squared_error,
    'l1_l2': l1_l2,
    # if really a wedge should be used then there is a problem here.
    # Bcz the wedge is not accessible in this file, but the CUSTOM_OBJECTS
    # is not accessible in unet.py. A solution might be to really serialize
    # the CUSTOM_OBJECTS
    'l2_on_wedge': l2_on_wedge_factory(None),
    'my_psnr': my_psnr
}
