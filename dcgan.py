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


def generator(input_tensor, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        net = layers.stack(input_tensor, layers.conv2d_transpose, [
            [ngf*8, [4, 4], 2, 'VALID', 'NHWC', tf.nn.relu, layers.batch_norm, {'reuse':reuse, 'scope':'generator'}],
            [ngf*4, [4, 4], 2,  'SAME', 'NHWC', tf.nn.relu, layers.batch_norm],
            [ngf*2, [4, 4], 2,  'SAME', 'NHWC', tf.nn.relu, layers.batch_norm],
            [ngf*1, [4, 4], 2,  'SAME', 'NHWC', tf.nn.relu, layers.batch_norm],
            [   nc, [4, 4], 2,  'SAME', 'NHWC', tf.nn.tanh, layers.batch_norm]
        ])
    return net


def discriminator(input_tensor, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        net = layers.stack(input_tensor, layers.conv2d, [
            [ndf*1, [4, 4], 2, 'SAME', None, 1, lrelu],
            [ndf*2, [4, 4], 2, 'SAME', None, 1, lrelu],
            [ndf*4, [4, 4], 2, 'SAME', None, 1, lrelu],
            [ndf*8, [4, 4], 2, 'SAME', None, 1, lrelu],
            [    1, [4, 4], 2, 'SAME', None, 1, tf.nn.sigmoid]
        ], normalizer_fn=layers.batch_norm)
    return net


def binary_cross_entropy(op1, op2, name=None):
    return tf.reduce_mean(op1 * tf.log(op2) + (1-op1)*tf.log(1-op2), name=name)


def main(_):

    batch_size = 64

    noise = tf.random_uniform((batch_size, 1, 1, nc), -1, 1, name='noise')
    pos_labels = tf.constant(1, shape=(batch_size, 1, 1, 1), dtype=tf.float32)
    neg_labels = tf.constant(0, shape=(batch_size, 1, 1, 1), dtype=tf.float32)
    # real_images = tf.placeholder(tf.float32, (None, 64, 64, 3))
    real_images = create_celeba_pipeline('/mnt/DataBlock/CelebA/Img/img_align_celeba.tfrecords')

    g_d_1_op = discriminator(generator(noise))                          # For training generator
    g_d_2_op = discriminator(generator(noise, reuse=True), reuse=True)  # for training discriminator with fake data
    d_op = discriminator(real_images, reuse=True)                       # for training discriminator with real data

    gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    g_d_1_loss = binary_cross_entropy(g_d_1_op, pos_labels, name='g_d_1_loss')
    g_d_2_loss = binary_cross_entropy(g_d_2_op, neg_labels, name='g_d_2_loss')
    d_loss = binary_cross_entropy(d_op, pos_labels, 'd_loss')

    g_train_op = tf.train.AdamOptimizer(0.0002, 0.5).minimize(g_d_1_loss, var_list=gen_var, name='g_train')
    d_real_train_op = tf.train.AdamOptimizer(0.0002, 0.5).minimize(d_loss, var_list=dis_var, name='real_d_train')
    d_fake_train_op = tf.train.AdamOptimizer(0.0002, 0.5).minimize(g_d_2_loss, var_list=dis_var, name='fake_d_train')

    with tf.Session() as sess:

        # Save the graph
        writer = tf.summary.FileWriter(logdir=FLAGS.log_dir, graph=tf.get_default_graph())

        # 1. Training discriminator
        # 1.1 Train with real(feed:image, update_variable:discriminator)
        sess.run(d_real_train_op)

        # 1.2 Train with fake(feed:noise, update_variable:discriminator)
        sess.run(d_fake_train_op)

        # 2. Training generator(feed:noise, update_variable:generator)
        sess.run(g_train_op)


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
