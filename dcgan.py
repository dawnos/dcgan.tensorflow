import argparse
import sys

import tensorflow as tf
from tensorflow.contrib import layers
from celeba import create_celeba_pipeline


nz = 100
ngf = 64
ndf = 64
nc = 3


def lrelu(x, alpha=0.2, name="lrelu"):
    return tf.maximum(x, alpha * x, name=name)


def generator(input_tensor, name='Generator', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        net = layers.stack(input_tensor, layers.conv2d_transpose, [
            [ngf*8, [4, 4], 2, 'VALID', 'NHWC', tf.nn.relu],
            [ngf*4, [4, 4], 2,  'SAME', 'NHWC', tf.nn.relu],
            [ngf*2, [4, 4], 2,  'SAME', 'NHWC', tf.nn.relu],
            [ngf*1, [4, 4], 2,  'SAME', 'NHWC', tf.nn.relu],
            [   nc, [4, 4], 2,  'SAME', 'NHWC', tf.nn.tanh]
        ], normalizer_fn=layers.batch_norm)
    return net


def discriminator(input_tensor, name='Discriminator', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        net = layers.stack(input_tensor, layers.conv2d, [
            [ndf*1, [4, 4], 2, 'SAME', None, 1, lrelu],
            [ndf*2, [4, 4], 2, 'SAME', None, 1, lrelu],
            [ndf*4, [4, 4], 2, 'SAME', None, 1, lrelu],
            [ndf*8, [4, 4], 2, 'SAME', None, 1, lrelu],
            [    1, [4, 4], 2, 'SAME', None, 1, tf.nn.sigmoid]
        ], normalizer_fn=layers.batch_norm)
    return net


def binary_cross_entropy(op1, op2, name='BinaryCrossEntropy'):
    with tf.variable_scope(name):
        return tf.reduce_mean(op1 * tf.log(op2) + (1-op1)*tf.log(1-op2))


def main(_):

    batch_size = 64

    noise = tf.random_uniform((batch_size, 1, 1, nc), -1, 1, name='Noise')
    pos_labels = tf.constant(1, shape=(batch_size, 1, 1, 1), dtype=tf.float32, name='Ones')
    neg_labels = tf.constant(0, shape=(batch_size, 1, 1, 1), dtype=tf.float32, name='Zeros')
    real_images = create_celeba_pipeline('/mnt/DataBlock/CelebA/Img/img_align_celeba.tfrecords', name='ImagePipeline')
    state = tf.placeholder(tf.int32, name='State')

    g_op = generator(noise, name='Generator')
    d_input_op = tf.cond(
        tf.equal(state, tf.constant(0), 'State0'), lambda: real_images, lambda: g_op, name='DiscriminatorInputSwitch')
    d_op = discriminator(d_input_op, name='Discriminator')
    label_op = tf.cond(tf.equal(state, tf.constant(1), 'State1'), lambda: neg_labels, lambda: pos_labels, name='Label')
    loss = binary_cross_entropy(d_op, label_op, name='loss')

    gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
    dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

    g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'Generator')
    d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'Discriminator')

    with tf.control_dependencies(g_update_ops):
        g_train_op = tf.train.AdamOptimizer(0.0002, 0.5).minimize(loss, var_list=gen_var, name='GTrain')

    with tf.control_dependencies(d_update_ops):
        d_train_op = tf.train.AdamOptimizer(0.0002, 0.5).minimize(loss, var_list=dis_var, name='DTrain')

    with tf.Session() as sess:

        # Save the graph
        writer = tf.summary.FileWriter(logdir=FLAGS.log_dir, graph=tf.get_default_graph())

        exit()
        # 1. Training discriminator
        # 1.1 Train with real(feed:image, update_variable:discriminator)    --> step 0
        sess.run(d_train_op, feed_dict={state:0})

        # 1.2 Train with fake(feed:noise, update_variable:discriminator)    --> step 1
        sess.run(d_train_op, feed_dict={state:1})

        # 2. Training generator(feed:noise, update_variable:generator)      -->step 2
        sess.run(g_train_op, feed_dict={state:2})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, uses fake data for unit testing.')

    parser.add_argument('--epoch', type=int, default=25,
                        help='Number of steps to run trainer.')

    parser.add_argument('--data_dir', type=str, default='/mnt/DataBlock/CelebA/Img',
                        help='Directory for storing input data')

    parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/dcgan/log',
                        help='Log directory')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
