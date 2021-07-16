"""Partially learned gradient descent scheme for ellipses."""
 
import os
import sys
import adler
#adler.util.gpu.setup_one_gpu()
 
from adler.tensorflow import prelu, cosine_decay, reference_unet, psnr
 
import tensorflow as tf
import numpy as np
import odl
import odl.contrib.tensorflow
 
 
 
def random_ellipse(interior=False):
    if interior:
        x_0 = np.random.rand() - 0.5
        y_0 = np.random.rand() - 0.5
    else:
        x_0 = 2 * np.random.rand() - 1.0
        y_0 = 2 * np.random.rand() - 1.0
 
    magnitude = np.random.choice([-1, 1]) * (0.1 + np.random.exponential(0.2))
 
    return (magnitude,
            np.random.exponential() * 0.2, np.random.exponential() * 0.2,
            x_0, y_0,
            np.random.rand() * 2 * np.pi)
 
 
def random_phantom(spc, n_ellipse=50, interior=False):
    n = np.random.poisson(n_ellipse)
    ellipses = [random_ellipse(interior=interior) for _ in range(n)]
    return odl.phantom.ellipsoid_phantom(spc, ellipses)
 
 
def make_difference(space):
    minp = (np.random.rand(2) - 0.5) - 0.05
    maxp = minp + 0.1 + 0.1 * (np.random.rand(2) - 0.5)
    scale = 0.5 * space.domain.extent
    magnitude = 0.1
    return magnitude * odl.phantom.cuboid(space, scale * minp, scale * maxp)
  

power = float(sys.argv[1])
const_val = 10 ** power
print('Running with const_val={}'.format(const_val))
 
np.random.seed(0)
name = os.path.splitext(os.path.basename(__file__))[0] + '/' + str(const_val)
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
 
 
with tf.name_scope('optimizer'):
    # Learning rate
    global_step = tf.Variable(0, trainable=False)
    maximum_steps = 100001
    starter_learning_rate = 3e-4
    learning_rate = cosine_decay(starter_learning_rate,
                                 global_step,
                                 maximum_steps,
                                 name='learning_rate')
 
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt_func = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                          beta2=0.99)
 
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1)
        optimizer = opt_func.apply_gradients(zip(grads, tvars),
                                             global_step=global_step)
 
 
# Summaries
# tensorboard --logdir=...
 
with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('loss_tomography', loss_tomography)
    tf.summary.scalar('loss_difference', loss_difference)
    tf.summary.scalar('psnr1', psnr(recon1, x_true1))
    tf.summary.scalar('psnr2', psnr(recon2, x_true2))
 
    tf.summary.image('recon1', recon1)
    tf.summary.image('recon2', recon2)
    tf.summary.image('x_true1', x_true1)
    tf.summary.image('x_true2', x_true2)
    tf.summary.image('difference', difference_result)
    tf.summary.image('difference_true', (x_true1 - x_true2))
 
    merged_summary = tf.summary.merge_all()
    test_summary_writer, train_summary_writer = adler.tensorflow.util.summary_writers(name, cleanup=True)
 
# Initialize all TF variables
sess.run(tf.global_variables_initializer())
 
# Add op to save and restore
saver = tf.train.Saver()
 
# Generate validation data
y_arr_validate1, x_true_arr_validate1, y_arr_validate2, x_true_arr_validate2 = generate_data(validation=True)
 
if 0:
    saver.restore(sess,
                  adler.tensorflow.util.default_checkpoint_path(name))
 
# Train the network
for i in range(0, maximum_steps):
    if i%10 == 0:
        y_arr1, x_true_arr1, y_arr2, x_true_arr2 = generate_data()
 
    _, merged_summary_result_train, global_step_result = sess.run([optimizer, merged_summary, global_step],
                              feed_dict={x_true1: x_true_arr1,
                                         y_rt1: y_arr1,
                                         x_true2: x_true_arr2,
                                         y_rt2: y_arr2,
                                         is_training: True,
                                         const: const_val})
 
    if i>0 and i%10 == 0:
        loss_result, merged_summary_result, global_step_result = sess.run([loss, merged_summary, global_step],
                              feed_dict={x_true1: x_true_arr_validate1,
                                         y_rt1: y_arr_validate1,
                                         x_true2: x_true_arr_validate2,
                                         y_rt2: y_arr_validate2,
                                         is_training: False,
                                         const: const_val})
 
        train_summary_writer.add_summary(merged_summary_result_train, global_step_result)
        test_summary_writer.add_summary(merged_summary_result, global_step_result)
 
        print('iter={}, loss={}'.format(global_step_result, loss_result))
 
    if i>0 and i%1000 == 0:
        saver.save(sess,
                   adler.tensorflow.util.default_checkpoint_path(name))