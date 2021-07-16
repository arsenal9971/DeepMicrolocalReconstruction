"""Partially learned gradient descent scheme for ellipses."""

import os
import adler
adler.util.gpu.setup_one_gpu()

from adler.tensorflow import prelu, cosine_decay, reference_unet
from adler.odl.phantom import random_phantom

import tensorflow as tf
import numpy as np
import odl
import odl.contrib.tensorflow
import scipy.ndimage



def make_difference(space):
    minp = (np.random.rand(2) - 0.5) - 0.05
    maxp = minp + 0.1 + 0.1 * (np.random.rand(2) - 0.5)
    scale = 0.5 * space.domain.extent
    magnitude = 0.1
    return magnitude * odl.phantom.cuboid(space, scale * minp, scale * maxp)


np.random.seed(0)
sess = tf.InteractiveSession()

# Create ODL data structures
size = 128
space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float32')

geometry = odl.tomo.parallel_beam_geometry(space, num_angles=30)
operator = odl.tomo.RayTransform(space, geometry)

# Ensure operator has fixed operator norm for scale invariance
opnorm = odl.power_method_opnorm(operator)
operator = (1 / opnorm) * operator

# Create tensorflow layer from odl operator
odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(operator,
                                                          'RayTransform')
odl_op_layer_adjoint = odl.contrib.tensorflow.as_tensorflow_layer(operator.adjoint,
                                                                  'RayTransformAdjoint')

# User selected paramters
n_data = 5
n_iter = 10
n_primal = 5
n_dual = 5

def generate_data(validation=False):
    """Generate a set of random data."""
    n_generate = 1 if validation else n_data

    y_arr1 = np.empty((n_generate, operator.range.shape[0], operator.range.shape[1], 1), dtype='float32')
    x_true_arr1 = np.empty((n_generate, space.shape[0], space.shape[1], 1), dtype='float32')
    y_arr2 = np.empty((n_generate, operator.range.shape[0], operator.range.shape[1], 1), dtype='float32')
    x_true_arr2 = np.empty((n_generate, space.shape[0], space.shape[1], 1), dtype='float32')

    for i in range(n_generate):
        if validation:
            phantom1 = odl.phantom.shepp_logan(space, True)
        else:
            phantom1 = random_phantom(space)
        
        phantom2 = phantom1 + make_difference(space)
            
            
        data1 = operator(phantom1)
        noisy_data1 = data1 + odl.phantom.white_noise(operator.range) * np.mean(np.abs(data1)) * 0.05
        data2 = operator(phantom2)
        noisy_data2 = data2 + odl.phantom.white_noise(operator.range) * np.mean(np.abs(data2)) * 0.05

        x_true_arr1[i, ..., 0] = phantom1
        y_arr1[i, ..., 0] = noisy_data1
        x_true_arr2[i, ..., 0] = phantom2
        y_arr2[i, ..., 0] = noisy_data2

    return y_arr1, x_true_arr1, y_arr2, x_true_arr2


with tf.name_scope('placeholders'):
    x_true1 = tf.placeholder(tf.float32, shape=[None, size, size, 1], name="x_true1")
    y_rt1 = tf.placeholder(tf.float32, shape=[None, operator.range.shape[0], operator.range.shape[1], 1], name="y_rt1")
    
    x_true2 = tf.placeholder(tf.float32, shape=[None, size, size, 1], name="x_true2")
    y_rt2 = tf.placeholder(tf.float32, shape=[None, operator.range.shape[0], operator.range.shape[1], 1], name="y_rt2")
    
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
    const = tf.placeholder(tf.float32, shape=(), name='const')


def apply_conv(x, filters=32):
    return tf.layers.conv2d(x, filters=filters, kernel_size=3, padding='SAME',
                            kernel_initializer=tf.contrib.layers.xavier_initializer())


def learned_primal_dual(data, reuse):
    with tf.variable_scope('learned_primal_dual', reuse=reuse):
        with tf.name_scope('initial_values'):
            primal = tf.concat([tf.zeros_like(x_true1)] * n_primal, axis=-1)
            dual = tf.concat([tf.zeros_like(data)] * n_dual, axis=-1)
    
        for i in range(n_iter):
            with tf.variable_scope('dual_iterate_{}'.format(i)):
                evalop = odl_op_layer(primal[..., 1:2])
                update = tf.concat([dual, evalop, data], axis=-1)
    
                update = prelu(apply_conv(update), name='prelu_1')
                update = prelu(apply_conv(update), name='prelu_2')
                update = apply_conv(update, filters=n_dual)
                dual = dual + update
    
            with tf.variable_scope('primal_iterate_{}'.format(i)):
                evalop = odl_op_layer_adjoint(dual[..., 0:1])
                update = tf.concat([primal, evalop], axis=-1)
    
                update = prelu(apply_conv(update), name='prelu_1')
                update = prelu(apply_conv(update), name='prelu_2')
                update = apply_conv(update, filters=n_primal)
                primal = primal + update
    
        return primal[..., 0:1]


with tf.name_scope('tomography'):
    recon1 = learned_primal_dual(y_rt1, reuse=False)
    recon2 = learned_primal_dual(y_rt2, reuse=True)


with tf.name_scope('edge_detect'):
    recons = tf.concat([recon1, recon2], axis=-1)
    difference_update = reference_unet(recons, 1,
                                 ndim=2,
                                 features=64,
                                 keep_prob=1.0,
                                 use_batch_norm=False,
                                 activation='relu',
                                 is_training=is_training,
                                 name='edge_result')
    
    difference_result = (recon1 - recon2) + difference_update


with tf.name_scope('loss'):
    loss_tomography = (tf.reduce_mean((recon1 - x_true1) ** 2) +
                       tf.reduce_mean((recon2 - x_true2) ** 2))

    loss_difference = tf.reduce_mean((difference_result - (x_true1 - x_true2)) ** 2)

    loss = loss_tomography + const * loss_difference
    

# Add op to save and restore
saver = tf.train.Saver()

# Generate validation data
y_arr_validate1, x_true_arr_validate1, y_arr_validate2, x_true_arr_validate2 = generate_data(validation=True)
import matplotlib.pyplot as plt
space.element(x_true_arr_validate1).show(clim=[0.1, 0.4], saveto='results/difference/true1.png')
space.element(x_true_arr_validate2).show(clim=[0.1, 0.4], saveto='results/difference/true2.png')
space.element(x_true_arr_validate2 - x_true_arr_validate1).show(clim=[0.0, 0.1], saveto='results/difference/difference_true.png')

for const_val in [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]:

    base_name = 'learned_primal_dual_difference_detect_parameter_search'
    name = base_name + '/' + str(const_val)

    saver.restore(sess,
                  adler.tensorflow.util.default_checkpoint_path(name))

    loss_result, loss_tomography_result, loss_difference_result, recon1_result, recon2_result, difference_result_result = sess.run([loss, loss_tomography, loss_difference, recon1, recon2, difference_result],
                          feed_dict={x_true1: x_true_arr_validate1,
                                     y_rt1: y_arr_validate1,
                                     x_true2: x_true_arr_validate2,
                                     y_rt2: y_arr_validate2,
                                     is_training: False,
                                     const: const_val})
        
    print('const= {}, loss_tomo = {}, loss_difference = {}'.format(const_val, loss_tomography_result, loss_difference_result))

    space.element(recon1_result).show(clim=[0.1, 0.4], saveto='results/difference/recon1_{}.png'.format(const_val))
    space.element(recon2_result).show(clim=[0.1, 0.4], saveto='results/difference/recon2_{}.png'.format(const_val))
    space.element(-difference_result_result).show(clim=[0.0, 0.1], saveto='results/difference/difference_{}.png'.format(const_val))
    plt.close('all')