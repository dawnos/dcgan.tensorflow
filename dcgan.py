import argparse
import sys
import warnings

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.contrib import layers

import celeba


nz = 100
ngf = 64
ndf = 64
nc = 3


def lrelu(x, alpha=0.2, name="lrelu"):
    return tf.maximum(x, alpha * x, name=name)


def generator(input_tensor, name='generator', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        net = layers.stack(input_tensor, layers.conv2d_transpose, [
            [ngf*8, [4, 4], 2, 'VALID', 'NHWC', tf.nn.relu, layers.batch_norm],
            [ngf*4, [4, 4], 2,  'SAME', 'NHWC', tf.nn.relu, layers.batch_norm],
            [ngf*2, [4, 4], 2,  'SAME', 'NHWC', tf.nn.relu, layers.batch_norm],
            [ngf*1, [4, 4], 2,  'SAME', 'NHWC', tf.nn.relu, layers.batch_norm],
            [   nc, [4, 4], 2,  'SAME', 'NHWC', tf.nn.tanh]
        ])
    return net


def discriminator(input_tensor, name='discriminator', reuse=False):

    with tf.variable_scope(name, reuse=reuse):
        net = layers.stack(input_tensor, layers.conv2d, [
            [ndf*1, [4, 4], 2,  'SAME', 'NHWC', 1, lrelu],
            [ndf*2, [4, 4], 2,  'SAME', 'NHWC', 1, lrelu, layers.batch_norm],
            [ndf*4, [4, 4], 2,  'SAME', 'NHWC', 1, lrelu, layers.batch_norm],
            [ndf*8, [4, 4], 2,  'SAME', 'NHWC', 1, lrelu, layers.batch_norm],
            [    1, [4, 4], 2, 'VALID', 'NHWC', 1, None]
        ])

    return net


def to_rgb(op, name='to_rgb'):
    with tf.variable_scope(name):
        rgb = tf.cast((op+1)*127.5, tf.uint8)
        return rgb


def main(_):

    batch_size = 128
    epoches = 25
    data_count = 202613

    noise = tf.random_uniform((batch_size, 1, 1, nc), -1, 1, name='noise')
    pos_labels = tf.constant(1, shape=(batch_size, 1, 1, 1), dtype=tf.float32, name='ONES')
    neg_labels = tf.constant(0, shape=(batch_size, 1, 1, 1), dtype=tf.float32, name='ZEROS')
    state = tf.placeholder(tf.int32, name='state')
    if FLAGS.raw_input:
        print 'Using raw input'
        real_images = tf.placeholder(tf.float32, shape=(batch_size, 64, 64, 3))
        reader = celeba.Reader('/mnt/DataBlock/CelebA/Img/img_align_celeba', batch_size=batch_size)
    else:
        real_images = celeba.create_pipeline('/mnt/DataBlock/CelebA/Img/img_align_celeba.tfrecords',
                                             name='image_pipeline', batch_size=batch_size)
        reader = None

    g_op = generator(noise, name='generator')
    d_input_op = tf.cond(
        tf.equal(state, tf.constant(0), 'STATE0'), lambda: real_images, lambda: g_op, name='discriminator_input_switch')
    d_op = discriminator(d_input_op, name='discriminator')
    label_op = tf.cond(tf.equal(state, tf.constant(1), 'STATE1'), lambda: neg_labels, lambda: pos_labels, name='label')
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_op, labels=label_op), name='loss')

    gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    g_train_op = tf.train.AdamOptimizer(0.0002, 0.5).minimize(loss, var_list=gen_var, name='g_train')
    d_train_op = tf.train.AdamOptimizer(0.0002, 0.5).minimize(loss, var_list=dis_var, name='d_train')

    # Summaries
    generated_image_summary_op = tf.summary.image('generated_image', to_rgb(g_op))
    real_image_summary_op = tf.summary.image('real_image', to_rgb(real_images))
    g_loss_summary_op = tf.summary.scalar('g_loss', loss)
    d_loss_summary_op = tf.summary.scalar('d_loss', loss)
    g_merged_summaries = tf.summary.merge([generated_image_summary_op, g_loss_summary_op])
    d_merged_summaries = tf.summary.merge([real_image_summary_op, d_loss_summary_op])

    writer = tf.summary.FileWriter(logdir=FLAGS.log_dir, graph=tf.get_default_graph())
    saver = tf.train.Saver()

    with tf.Session() as sess:

        if not FLAGS.raw_input:
            tf.train.start_queue_runners(sess)
            print 'Queue runners started.'

        if FLAGS.debug:
            print 'Entering debug mode...'
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        tf.global_variables_initializer().run(session=sess)
        print 'Variables initialized'

        # Main
        epoch = 0.0
        global_step = 0
        while epoch < epoches:

            epoch += float(batch_size) / data_count

            # 1. Training discriminator
            # 1.1 Train with real: state 0
            if reader:
                _, summary, loss_val = sess.run([d_train_op, d_merged_summaries, loss],
                    feed_dict={state: 0, real_images: reader.next_batch()})
            else:
                _, summary, loss_val = sess.run([d_train_op, d_merged_summaries, loss], feed_dict={state: 0})
            writer.add_summary(summary, global_step)
            print 'step 0: loss=' + str(loss_val)
            global_step += 1

            # 1.2 Train with fake: state 1
            if reader:
                _, summary, loss_val = sess.run([d_train_op, d_merged_summaries, loss],
                    feed_dict={state: 1, real_images: reader.next_batch()})
            else:
                _, summary, loss_val = sess.run([d_train_op, d_merged_summaries, loss], feed_dict={state: 1})
            writer.add_summary(summary, global_step)
            print 'step 1: loss=' + str(loss_val)
            global_step += 1

            # 2. Training generator: state 2
            if reader:
                _, summary, loss_val = sess.run([g_train_op, g_merged_summaries, loss],
                    feed_dict={state: 2, real_images: reader.next_batch()})
            else:
                _, summary, loss_val = sess.run([g_train_op, g_merged_summaries, loss], feed_dict={state: 2})
            writer.add_summary(summary, global_step)
            print 'step 2: loss=' + str(loss_val)
            global_step += 1

            print 'Epoch:%.2f/%d' % (epoch, epoches)

            if global_step/3 % FLAGS.save_interval == 0:
                print 'Saving model...'
                saver.save(sess, FLAGS.log_dir + "/model.ckpt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int, default=25,
                        help='Number of steps to run trainer.')

    parser.add_argument('--data_dir', type=str, default='/mnt/DataBlock/CelebA/Img',
                        help='Directory for storing input data or tfrecord file')

    parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/dcgan/log',
                        help='Log directory')

    parser.add_argument("--debug", type=bool, default=False,
                        help="Use debugger to track down bad values during training")

    parser.add_argument("--raw_input", type=bool, default=False,
                        help="If true, read data from separate images; otherwise from tfrecord")

    parser.add_argument("--save_interval", type=int, default=500,
                        help="Save interval")

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.debug and not FLAGS.raw_input:
        warnings.warn('In debug mode only raw input are allowed. Changing to raw input.')
        FLAGS.raw_input = True

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
