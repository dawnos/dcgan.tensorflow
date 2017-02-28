import argparse
import sys
import numpy as np

import tensorflow as tf
from tensorflow.contrib import layers

# from keras.models import Sequential
# from keras.layers import Deconvolution2D, Activation, BatchNormalization, Convolution2D, LeakyReLU, Dense

nz = 100
ngf = 64
ndf = 64
nc = 3


def lrelu(x, alpha=0.2, name="lrelu"):
    return tf.maximum(x, alpha * x, name=name)


def generator(input_tensor, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        net =layers.stack(input_tensor, layers.conv2d_transpose, [
            [ngf*8, [4, 4], 2, 'VALID', 'NHWC', tf.nn.relu],
            [ngf*4, [4, 4], 2, 'SAME' , 'NHWC', tf.nn.relu],
            [ngf*2, [4, 4], 2, 'SAME' , 'NHWC', tf.nn.relu],
            [ngf*1, [4, 4], 2, 'SAME' , 'NHWC', tf.nn.relu],
            [nc   , [4, 4], 2, 'SAME' , 'NHWC', tf.nn.tanh]
        ], normalizer_fn=layers.batch_norm)
        # net = layers.conv2d_transpose(input_tensor, ngf * 8, padding='VALID', kernel_size=[4,4], stride=2, normalizer_fn=layers.batch_norm)
        # net = layers.stack(net,
        #                    layers.conv2d_transpose, [ngf * 4, ngf * 2, ngf * 1], padding='SAME', kernel_size=[4,4], stride=2, normalizer_fn=layers.batch_norm)
        # net = layers.conv2d_transpose(net, nc, padding='SAME', activation_fn=tf.nn.tanh, kernel_size=[4,4], stride=2, normalizer_fn=layers.batch_norm)
    return net


    # model = Sequential([
    #     Deconvolution2D(ngf * 8, 4, 4, (None, ngf * 8,  4,  4), subsample=(2, 2), border_mode='same', input_shape=(1, 1, nz)),
    #     BatchNormalization(),
    #     Activation('relu'),
    #
    #     Deconvolution2D(ngf * 4, 4, 4, (None, ngf * 4,  8,  8), subsample=(2, 2), border_mode='same'),
    #     BatchNormalization(),
    #     Activation('relu'),
    #
    #     Deconvolution2D(ngf * 2, 4, 4, (None, ngf * 2, 16, 16), subsample=(2, 2), border_mode='same'),
    #     BatchNormalization(),
    #     Activation('relu'),
    #
    #     Deconvolution2D(ngf * 1, 4, 4, (None, ngf * 1, 32, 32), subsample=(2, 2), border_mode='same'),
    #     BatchNormalization(),
    #     Activation('relu'),
    #
    #     Deconvolution2D(     nc, 4, 4, (None,      nc, 64, 64), subsample=(2, 2), border_mode='same'),
    #     Activation('tanh'),
    # ], 'generator')
    # return model(input_tensor)


def discriminator(input_tensor, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        net = layers.stack(input_tensor, layers.conv2d, [
            [ndf*1, [4, 4], 2, 'SAME', None, 1, lrelu],
            [ndf*2, [4, 4], 2, 'SAME', None, 1, lrelu],
            [ndf*4, [4, 4], 2, 'SAME', None, 1, lrelu],
            [ndf*8, [4, 4], 2, 'SAME', None, 1, lrelu],
            [1    , [4, 4], 2, 'SAME', None, 1, tf.nn.sigmoid]
        ], normalizer_fn=layers.batch_norm)

        # net = layers.stack(input_tensor, layers.conv2d,
        #                    [ndf * 1, ndf * 2, ndf * 4, ndf * 8],
        #                    kernel_size=[4, 4], normalizer_fn=layers.batch_norm)
        # net = layers.conv2d(net, 1, kernel_size=[4, 4], normalizer_fn=layers.batch_norm, activation_fn=tf.nn.sigmoid)
    return net

    # with tf.variable_scope('discriminator', reuse=reuse):
    # model = Sequential([
    #     Convolution2D(ndf, 4, 4, border_mode='same', input_shape=(64, 64, nc)),
    #     LeakyReLU(0.2),
    #
    #     Convolution2D(ndf * 2, 4, 4, border_mode='same'),
    #     BatchNormalization(),
    #     LeakyReLU(0.2),
    #
    #     Convolution2D(ndf * 4, 4, 4, border_mode='same'),
    #     BatchNormalization(),
    #     LeakyReLU(0.2),
    #
    #     Convolution2D(ndf * 8, 4, 4, border_mode='same'),
    #     Activation('sigmoid')
    # ], 'discriminator')
    # # ans = model(input_tensor)
    # # ans = Convolution2D(ndf, 4, 4, border_mode='same', input_shape=(64, 64, nc))(input_tensor)
    # # ans = tf.layers.conv2d(input_tensor, filter=5, kernel_size=[5, 5], padding='SAME')
    # # ans = slim.conv2d(input_tensor, 32, [5, 5])
    # # return ans
    # return model


def binary_cross_entropy(op1, op2):
    return tf.reduce_mean(op1 * tf.log(op2) + (1-op1)*tf.log(1-op2))


def main(_):

    batch_size = 64

    # noise = tf.placeholder(tf.float32, (batch_size, 1, 1, nz))
    noise = tf.random_uniform((batch_size, 1, 1, nc), -1, 1, name='noise')
    real_images = tf.placeholder(tf.float32, (None, 64, 64, 3))
    pos_labels = tf.constant(1, shape=(batch_size, 1, 1, 1))
    neg_labels = tf.constant(0, shape=(batch_size, 1, 1, 1))
    # labels = tf.placeholder(tf.float32, (batch_size, 1, 1, 1))

    g_d_1_op = discriminator(generator(noise))              # For training generator
    g_d_2_op = discriminator(generator(noise, reuse=True))  # for training discriminator with fake data
    d_op = discriminator(real_images, reuse=True)           # for training discriminator with real data

    gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    g_d_1_loss = binary_cross_entropy(g_d_1_op, pos_labels)
    g_d_2_loss = binary_cross_entropy(g_d_2_op, neg_labels)
    # tf.reduce_mean(g_d_1_op * tf.log(labels) + (1 - g_d_1_op) * tf.log(1 - labels), name='g_d_loss')
    d_loss = binary_cross_entropy(d_op, pos_labels)
    # tf.reduce_mean(d_op * tf.log(labels) + (1 - d_op) * tf.log(1 - labels), name='d_loss')

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

        # all_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # exit()

        # merged = tf.summary.merge_all()
        # train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',
        #                                      sess.graph)
        # test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
        # tf.global_variables_initializer().run()


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
