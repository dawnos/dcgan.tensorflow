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
            [ngf*8, [4, 4], 2, 'VALID', 'NHWC', tf.nn.relu],
            [ngf*4, [4, 4], 2,  'SAME', 'NHWC', tf.nn.relu],
            [ngf*2, [4, 4], 2,  'SAME', 'NHWC', tf.nn.relu],
            [ngf*1, [4, 4], 2,  'SAME', 'NHWC', tf.nn.relu],
            [   nc, [4, 4], 2,  'SAME', 'NHWC', tf.nn.tanh]
        ], normalizer_fn=layers.batch_norm)
    return net


def discriminator(input_tensor, name='discriminator', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        '''
        net = layers.stack(input_tensor, layers.conv2d, [
            [ndf*1, [4, 4], 2, 'SAME', None, 1, lrelu],
            [ndf*2, [4, 4], 2, 'SAME', None, 1, lrelu],
            [ndf*4, [4, 4], 2, 'SAME', None, 1, lrelu],
            [ndf*8, [4, 4], 2, 'SAME', None, 1, lrelu],
            [    1, [4, 4], 2, 'VALID', None, 1, None]
        ], normalizer_fn=layers.batch_norm)
        '''

        net = layers.conv2d(input_tensor, ndf * 1, [4, 4], 2, 'SAME', activation_fn=lrelu)
        net = layers.conv2d(net, ndf * 2, [4, 4], 2, 'SAME', activation_fn=lrelu, normalizer_fn=layers.batch_norm)
        net = layers.conv2d(net, ndf * 4, [4, 4], 2, 'SAME', activation_fn=lrelu, normalizer_fn=layers.batch_norm)
        net = layers.conv2d(net, ndf * 8, [4, 4], 2, 'SAME', activation_fn=lrelu, normalizer_fn=layers.batch_norm)
        net = layers.conv2d(net, 1, [4, 4], 2, 'VALID', activation_fn=None)

    return net


def binary_cross_entropy(op1, op2, name='binary_cross_entropy'):
    with tf.variable_scope(name):
        return tf.reduce_mean(op1 * tf.log(op2) + (1-op1)*tf.log(1-op2))


def to_rgb(op, name='to_rgb'):
    with tf.variable_scope(name):
        rgb = tf.cast((op+1)*127.5, tf.uint8)
        return rgb


def main(_):

    batch_size = 128
    epoches = 25
    data_count = 202613

    noise = tf.random_uniform((batch_size, 1, 1, nc), -1, 1, name='noise')
    pos_labels = tf.constant(1, shape=(batch_size, 1, 1, 1), dtype=tf.float32, name='ones')
    neg_labels = tf.constant(0, shape=(batch_size, 1, 1, 1), dtype=tf.float32, name='zeros')
    state = tf.placeholder(tf.int32, name='state')
    if FLAGS.raw_input:
        print 'Using raw input'
        real_images = tf.placeholder(tf.float32, shape=(batch_size, 64, 64, 3))
        reader = celeba.Reader('/mnt/DataBlock/CelebA/Img/img_align_celeba', batch_size=batch_size)
    else:
        real_images = celeba.create_pipeline('/mnt/DataBlock/CelebA/Img/img_align_celeba.tfrecords',
                                             name='image_pipeline', batch_size=batch_size)

    g_op = generator(noise, name='generator')

    '''
    d_input_op_1 = tf.get_variable('d_input_op_1', dtype=tf.float32, shape=(batch_size, 64, 64, 3))
    def fn1():
        with tf.control_dependencies([tf.assign(d_input_op_1, real_images)]):
            return tf.identity(d_input_op_1)
    def fn2():
        with tf.control_dependencies([tf.assign(d_input_op, g_op)]):
            return tf.identity(d_input_op)
    d_input_op = tf.cond(tf.equal(state, tf.constant(0), 'STATE0'), fn1, lambda: tf.identity(d_input_op_1))
    '''

    d_input_op = tf.cond(tf.equal(state, tf.constant(0), 'STATE0'), lambda: real_images, lambda: g_op, name='discriminator_input_switch')

    d_op = discriminator(d_input_op, name='discriminator')
    tf.summary.scalar('tmp', d_op)
    label_op = tf.cond(tf.equal(state, tf.constant(1), 'STATE1'), lambda: neg_labels, lambda: pos_labels, name='label')
    # loss = binary_cross_entropy(d_op, label_op, name='loss')
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_op, labels=label_op), name='loss')

    gen_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    dis_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    # g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'generator')
    # d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'discriminator')

    # with tf.control_dependencies(g_update_ops):
    g_train_op = tf.train.AdamOptimizer(0.0002, 0.5).minimize(loss, var_list=gen_var, name='g_train')

    # with tf.control_dependencies(d_update_ops):
    d_train_op = tf.train.AdamOptimizer(0.0002, 0.5).minimize(loss, var_list=dis_var, name='d_train')

    # with tf.variable_scope('discriminator/Stack/convolution_5/BatchNorm', reuse=True):
    #     bn = tf.get_variable('moving_mean')
    # with tf.variable_scope('', reuse=True):
    #     bn = tf.get_variable('loss')

    # Summaries
    generated_image_summary_op = tf.summary.image('generated_image', to_rgb(g_op))
    real_image_summary_op = tf.summary.image('real_image', to_rgb(real_images))
    g_loss_summary_op = tf.summary.scalar('g_loss', loss)
    d_loss_summary_op = tf.summary.scalar('d_loss', loss)
    # bn_summary_op = tf.summary.scalar('bn11', bn)
    g_merged_summaries = tf.summary.merge([generated_image_summary_op, g_loss_summary_op])
    d_merged_summaries = tf.summary.merge([real_image_summary_op, d_loss_summary_op])

    with tf.Session() as sess:

        # Save the graph
        writer = tf.summary.FileWriter(logdir=FLAGS.log_dir, graph=tf.get_default_graph())
        print 'Graph written'

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
            global_step += 1

            # 1. Training discriminator
            # 1.1 Train with real(feed:image, update_variable:discriminator)    --> state 0
            if FLAGS.raw_input:
                _, summary, loss_val = sess.run([d_train_op, d_merged_summaries, loss],
                                                feed_dict={state: 0, real_images: reader.next_batch()})
            else:
                _, summary, loss_val = sess.run([d_train_op, d_merged_summaries, loss], feed_dict={state: 0})
            writer.add_summary(summary, global_step)
            print 'step 0: loss=' + str(loss_val)

            global_step += 1

            # 1.2 Train with fake(feed:noise, update_variable:discriminator)    --> state 1
            if FLAGS.raw_input:
                _, summary, loss_val = sess.run([d_train_op, d_merged_summaries, loss],
                                                feed_dict={state: 0, real_images:reader.next_batch()})
            else:
                _, summary, loss_val = sess.run([d_train_op, d_merged_summaries, loss], feed_dict={state: 1})
            writer.add_summary(summary, global_step)
            print 'step 1: loss=' + str(loss_val)

            # 2. Training generator(feed:noise, update_variable:generator)      -->state 2
            if FLAGS.raw_input:
                _, summary, loss_val = sess.run([d_train_op, d_merged_summaries, loss],
                                                feed_dict={state: 0, real_images:reader.next_batch()})
            else:
                _, summary, loss_val = sess.run([g_train_op, g_merged_summaries, loss], feed_dict={state: 2})
            writer.add_summary(summary, global_step)
            print 'step 2: loss=' + str(loss_val)

            print 'Epoch:%.2f/%d' % (epoch, epoches)

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

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.debug and not FLAGS.raw_input:
        warnings.warn('In debug mode only raw input are allowed. Changing to raw input.')
        FLAGS.raw_input = True

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
