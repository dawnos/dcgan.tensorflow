# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts CelebA data to TFRecords file format."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
from scipy.misc import imread, imresize
import tensorflow as tf
import progressbar

FLAGS = None


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, width, height, channel):

    num_examples = len(data_set)

    rows = height
    cols = width
    depth = channel
    bar = progressbar.ProgressBar(num_examples, widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                            progressbar.Percentage(), ' ',
                                            progressbar.ETA()])
    bar.start()

    filename = FLAGS.directory + '.tfrecords'
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        bar.update(index)
        image_raw = imresize(imread(os.path.join(FLAGS.directory, data_set[index])), (width, height))
        if FLAGS.preprocess:
            image_raw = (image_raw - 127.5) / 127.5
        image_raw = image_raw.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


def main(unused_argv):
    filelist = os.listdir(FLAGS.directory)

    # Convert to Examples and write the result to TFRecords.
    convert_to(filelist, 64, 64, 3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        default='/mnt/DataBlock/CelebA/Img/img_align_celeba',
        help='Directory to download data files and write the converted result'
    )

    parser.add_argument(
        '--preprocess',
        type=bool,
        default=False,
        help='Whether to preprocess'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
