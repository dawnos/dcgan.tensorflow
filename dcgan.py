import argparse
import sys

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.contrib import layers

from celeba import create_celeba_pipeline


nz = 100
ngf = 64
ndf = 64
nc = 3


def lrelu(x, alpha=0.2, name="lrelu"):
    return tf.maximum(x, alpha * x, name=name)


def generator(input_tensor, name='generator', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        net = layers.stack(input_tensor, layers.conv2d_transpose, [
            [ngf*8, [4, 4], 2, 'VALID', 'NHWC', tf.nn.relu],
            [ngf*4, [4, 4], 2,  'SAME', 'NHWC', tf.nn.relu],
            [ngf*2, [4, 4], 2,  'SAME', 'NHWC', tf.nn.relu],
            [ngf*1, [4, 4], 2,  'SAME', 'NHWC', tf.nn.relu],
            [   nc, [4, 4], 2,  'SAME', 'NHWC', tf.nn.tanh]
        ], normalizer_fn=layers.batch_norm)
    return net


def discriminator(input_tensor, name='discriminator', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        net = layers.stack(input_tensor, layers.conv2d, [
            [ndf*1, [4, 4], 2, 'SAME', None, 1, lrelu],
            [ndf*2, [4, 4], 2, 'SAME', None, 1, lrelu],
            [ndf*4, [4, 4], 2, 'SAME', None, 1, lrelu],
            [ndf*8, [4, 4], 2, 'SAME', None, 1, lrelu],
            [    1, [4, 4], 2, 'SAME', None, 1, tf.nn.sigmoid]
        ], normalizer_fn=layers.batch_norm)
    return net


def binary_cross_entropy(op1, op2, name='binary_cross_entropy'):
    with tf.variable_scope(name):
        return tf.reduce_mean(op1 * tf.log(op2) + (1-op1)*tf.log(1-op2))


def to_rgb(op, name='to_rgb'):
    with tf.variable_scope(name):
        rgb = tf.cast((op+1)*127.5, tf.uint8)
        return rgb


def main(_):

    batch_size = 64
    epoches = 25
    data_count = 202613

    noise = tf.random_uniform((batch_size, 1, 1, nc), -1, 1, name='noise')
    pos_labels = tf.constant(1, shape=(batch_size, 1, 1, 1), dtype=tf.float32, name='ones')
    neg_labels = tf.constant(0, shape=(batch_size, 1, 1, 1), dtype=tf.float32, name='zeros')
    real_images, tmp = create_celeba_pipeline('/mnt/DataBlock/CelebA/Img/img_align_celeba.tfrecords', name='image_pipeline')
    state = tf.placeholder(tf.int32, name='state')

    g_op = generator(noise, name='generator')
    d_input_op = tf.cond(
        tf.equal(state, tf.constant(0), 'STATE0'), lambda: real_images, lambda: g_op, name='discriminator_input_switch')
    d_op = discriminator(d_input_op, name='discriminator')
    label_op = tf.cond(tf.equal(state, tf.constant(1), 'STATE1'), lambda: neg_labels, lambda: pos_labels, name='label')
    loss = binary_cross_entropy(d_op, label_op, name='loss')

    gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    # g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')
    # d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')

    # with tf.control_dependencies(g_update_ops):
    g_train_op = tf.train.AdamOptimizer(0.0002, 0.5).minimize(loss, var_list=gen_var, name='g_train')

    # with tf.control_dependencies(d_update_ops):
    d_train_op = tf.train.AdamOptimizer(0.0002, 0.5).minimize(loss, var_list=dis_var, name='d_train')

    # Summaries
    generated_image_summary_op = tf.summary.image('generated_image', to_rgb(g_op))
    real_image_summary_op = tf.summary.image('real_image', to_rgb(real_images))
    g_loss_summary_op = tf.summary.scalar('g_loss', loss)
    d_loss_summary_op = tf.summary.scalar('d_loss', loss)
    g_merged_summaries = tf.summary.merge([generated_image_summary_op, real_image_summary_op, g_loss_summary_op])
    d_merged_summaries = tf.summary.merge([generated_image_summary_op, real_image_summary_op, d_loss_summary_op])
    # g_train_op = [g_train_op, g_merged_summaries]
    # d_train_op = [d_train_op, d_merged_summaries]

    with tf.Session() as sess:

        tf.train.start_queue_runners()

        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # Save the graph
        writer = tf.summary.FileWriter(logdir=FLAGS.log_dir, graph=tf.get_default_graph())
        print 'graph written'

        # exit()
        tf.global_variables_initializer().run()
        print 'global variables initialized'

        # Main
        epoch = 0.0
        global_step = 0
        while epoch < epoches:
            epoch += batch_size / data_count
            global_step += 1

            # 1. Training discriminator
            # 1.1 Train with real(feed:image, update_variable:discriminator)    --> state 0
            _, summary = sess.run([d_train_op, d_merged_summaries], feed_dict={state: 0})
            # ans = sess.run(tmp, feed_dict={state: 0})
            # print ans
            print 'step 0'
            writer.add_summary(summary, global_step)

            # 1.2 Train with fake(feed:noise, update_variable:discriminator)    --> state 1
            _, summary = sess.run([d_train_op, d_merged_summaries], feed_dict={state: 1})
            print 'step 1'
            writer.add_summary(summary, global_step)

            # 2. Training generator(feed:noise, update_variable:generator)      -->state 2
            _, summary = sess.run([g_train_op, g_merged_summaries], feed_dict={state: 2})
            print 'step 2'
            writer.add_summary(summary, global_step)

            print 'Epoch:' + str(epoch) + '/' + str(epoches)


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
    parser.add_argument("--debug", type=bool, default=False,
                        help="Use debugger to track down bad values during training")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
