"""Partially learned gradient descent scheme for ellipses."""

import os
import adler

from adler.tensorflow import prelu, cosine_decay, reference_unet

import tensorflow as tf
import numpy as np
import odl
import odl.contrib.tensorflow
import scipy.ndimage
import lcr_data


np.random.seed(1)
tf.set_random_seed(1)

c_values = [0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]
idx = 7 # int(sys.argv[1])
const = c_values[idx]


np.random.seed(0)
name = os.path.splitext(os.path.basename(__file__))[0] + '/' + str(const)

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

# edge detection
grad = odl.Gradient(space)
edge_detector = odl.PointwiseNorm(grad.range) * grad

# User selected paramters
n_data = 5
n_iter = 10
n_primal = 5
n_dual = 5

# User selected paramters
head = lcr_data.davids_head_density()
head_true = lcr_data.davids_head_materials() == 4

def generate_data(validation=False):
    """Generate a set of random data."""
    n_generate = 1 if validation else n_data

    y_arr_rt = np.empty((n_generate, operator.range.shape[0], operator.range.shape[1], 1), dtype='float32')
    x_tomo_true_arr = np.empty((n_generate, space.shape[0], space.shape[1], 1), dtype='float32')
    x_segment_true_arr = np.empty((n_generate, space.shape[0], space.shape[1], 1), dtype='float32')

    for i in range(n_generate):
        if validation:
            phantom_rt = head[..., 33]
            true = head_true[..., 33]
        else:
            idx = np.random.randint(64)

            if idx>=33:
                j = idx + 1
            else:
                j = idx
            phantom_rt = head[..., j]
            true = head_true[..., j]

        # Data augumentation
        angle = 20 * (np.random.rand() - 0.5)
        phantom_rt = scipy.ndimage.interpolation.rotate(phantom_rt, angle, reshape=False, order=1)
        true = np.round(scipy.ndimage.interpolation.rotate(true.astype(float), angle, reshape=False, order=1))

        roll_x = int(40 * (np.random.rand() - 0.5))
        roll_y = int(40 * (np.random.rand() - 0.5))
        phantom_rt = np.roll(phantom_rt, roll_x, axis=0)
        phantom_rt = np.roll(phantom_rt, roll_y, axis=1)
        true = np.roll(true, roll_x, axis=0)
        true = np.roll(true, roll_y, axis=1)

        phantom_rt = phantom_rt[::4, ::4]
        true = true[::4, ::4]

        data_rt = operator(phantom_rt)
        noisy_data_rt = data_rt + odl.phantom.white_noise(operator.range) * np.mean(np.abs(data_rt)) * 0.001

        x_tomo_true_arr[i, ..., 0] = phantom_rt
        x_segment_true_arr[i, ..., 0] = true
        y_arr_rt[i, ..., 0] = noisy_data_rt


    return y_arr_rt, x_tomo_true_arr, x_segment_true_arr


with tf.name_scope('placeholders'):
    x_true = tf.placeholder(tf.float32, shape=[None, size, size, 1], name="x_true")
    y_rt = tf.placeholder(tf.float32, shape=[None, operator.range.shape[0], operator.range.shape[1], 1], name="y_rt")
    edge_true = tf.placeholder(tf.float32, shape=[None, size, size, 1], name="edge_true")
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training')


def apply_conv(x, filters=32):
    return tf.layers.conv2d(x, filters=filters, kernel_size=3, padding='SAME',
                            kernel_initializer=tf.contrib.layers.xavier_initializer())


with tf.name_scope('tomography'):
    with tf.name_scope('initial_values'):
        primal = tf.concat([tf.zeros_like(x_true)] * n_primal, axis=-1)
        dual = tf.concat([tf.zeros_like(y_rt)] * n_dual, axis=-1)

    for i in range(n_iter):
        with tf.variable_scope('dual_iterate_{}'.format(i)):
            evalop = odl_op_layer(primal[..., 1:2])
            update = tf.concat([dual, evalop, y_rt], axis=-1)

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

    x_result = primal[..., 0:1]


with tf.name_scope('possibly_stop'):
    if const < 1.0:
        x_result_stop = tf.identity(x_result)
    else:
        x_result_stop = tf.stop_gradient(x_result)


with tf.name_scope('edge_detect'):
    edge_result = reference_unet(x_result_stop, 1,
                                 ndim=2,
                                 features=32,
                                 keep_prob=0.7,
                                 use_batch_norm=False,
                                 activation='relu',
                                 is_training=is_training,
                                 name='edge_result')


with tf.name_scope('loss'):
    residual = x_result - x_true
    squared_error = residual ** 2
    loss_tomography = tf.reduce_mean(squared_error)

    seg_error = tf.nn.sigmoid_cross_entropy_with_logits(labels=edge_true,
                                                        logits=edge_result)
    loss_seg = tf.reduce_mean(seg_error)

    if const < 1.0:
        loss = (1 - const) * loss_tomography + const * loss_seg
    else:
        loss = loss_tomography + loss_seg


def clip_by_global_norm(x, clip):
    norm = tf.sqrt(2 * tf.reduce_sum(tf.stack([tf.nn.l2_loss(xi) for xi in x])))

    clip_val = tf.maximum(tf.cast(clip, 'float32'), norm)

    return [xi / clip_val for xi in x]


with tf.name_scope('optimizer'):
    # Learning rate
    global_step = tf.Variable(0, trainable=False)
    maximum_steps = 100001
    starter_learning_rate = 1e-3
    learning_rate = cosine_decay(starter_learning_rate,
                                 global_step,
                                 maximum_steps,
                                 name='learning_rate')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt_func = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                          beta2=0.99)

        tvars = tf.trainable_variables()
        grads = clip_by_global_norm(tf.gradients(loss, tvars), 1)
        optimizer = opt_func.apply_gradients(zip(grads, tvars),
                                             global_step=global_step)


# Summaries
# tensorboard --logdir=...

with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('loss_tomography', loss_tomography)
    tf.summary.scalar('loss_seg', loss_seg)
    tf.summary.scalar('psnr', -10 * tf.log(loss_tomography) / tf.log(10.0))

    tf.summary.image('x_result', x_result)
    tf.summary.image('x_true', x_true)
    tf.summary.image('x_edge_result', tf.sigmoid(edge_result))
    tf.summary.image('x_edge_true', edge_true)
    tf.summary.image('squared_error', squared_error)
    tf.summary.image('residual', residual)

    merged_summary = tf.summary.merge_all()
    test_summary_writer, train_summary_writer = adler.tensorflow.util.summary_writers(name, cleanup=True)

# Initialize all TF variables
sess.run(tf.global_variables_initializer())

# Add op to save and restore
saver = tf.train.Saver()

# Generate validation data
y_arr_validate, x_true_arr_validate, edge_arr_validate = generate_data(validation=True)

if 0:
    saver.restore(sess,
                  adler.tensorflow.util.default_checkpoint_path(name))

# Train the network
for i in range(0, maximum_steps):
    if i%10 == 0:
        y_arr, x_true_arr, edge_arr = generate_data()

    _, merged_summary_result_train, global_step_result = sess.run([optimizer, merged_summary, global_step],
                              feed_dict={x_true: x_true_arr,
                                         y_rt: y_arr,
                                         edge_true: edge_arr,
                                         is_training: True})

    if i>0 and i%10 == 0:
        lt, ls, merged_summary_result, global_step_result = sess.run([loss_tomography, loss_seg, merged_summary, global_step],
                              feed_dict={x_true: x_true_arr_validate,
                                         y_rt: y_arr_validate,
                                         edge_true: edge_arr_validate,
                                         is_training: False})

        train_summary_writer.add_summary(merged_summary_result_train, global_step_result)
        test_summary_writer.add_summary(merged_summary_result, global_step_result)

        print('iter={}, C={}, loss_tomography={}, loss_seg={}'.format(global_step_result, const, lt, ls))

    if i>0 and i%1000 == 0:
        saver.save(sess,
                   adler.tensorflow.util.default_checkpoint_path(name))
