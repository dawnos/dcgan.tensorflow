from os import listdir
import random
from scipy.misc import imread, imresize
import numpy as np
import tensorflow as tf


def create_pipeline(filename, width=64, height=64, depth=3, batch_size=64, name=None):
    with tf.variable_scope(name):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        key, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
            })

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape(height * width * depth)
        image = tf.reshape(image, (height, width, depth))
        image = tf.cast(image, tf.float32)
        image = image / 127.5 - 1

        pipeline = tf.train.shuffle_batch(
            [image], batch_size=batch_size, num_threads=4,
            capacity=1000 + 3 * batch_size, min_after_dequeue=1000)

        return pipeline


class Reader:
    def __init__(self, path, width=64, height=64, batch_size=64, shuffle=True):
        self.file_list = listdir(path)
        self.path = path
        self.file_count = len(self.file_list)
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.shuffle = shuffle
        self.last_read = 0

    def next_batch(self):
        batch = np.ndarray([self.batch_size, self.height, self.width, 3])
        for i in xrange(0, self.batch_size):
            if self.shuffle:
                ind = random.randint(0, self.file_count - 1)
                filename = self.file_list[ind]
                image = imresize(imread(self.path + '/' + filename), (self.height, self.width))

                batch[i, :, :, :] = \
                    (image - 127.5) / 127.5
            else:
                raise NotImplementedError

        return batch


def test():
    reader = Reader("/mnt/DataBlock/CelebA/Img/img_align_celeba", 64, 64)
    for i in xrange(0, 10000):
        batch = reader.next_batch()
        print 'done'


if __name__ == "__main__":
    test()
