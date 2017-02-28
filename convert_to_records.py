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

FLAGS = None


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, width, height, channel):
  """Converts a dataset to tfrecords."""
  # images = data_set.images
  # labels = data_set.labels
  num_examples = len(data_set)

  # if images.shape[0] != num_examples:
  #   raise ValueError('Images size %d does not match label size %d.' %
  #                    (images.shape[0], num_examples))
  rows = height
  cols = width
  depth = channel

  filename = FLAGS.directory + '.tfrecords'
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    # image_raw = images[index].tostring()
    image_raw = imresize(imread(os.path.join(FLAGS.directory, data_set[index])), (width, height))
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
  # Get the data.
  data_sets = os.listdir(FLAGS.directory)

  # Convert to Examples and write the result to TFRecords.
  convert_to(data_sets, 64, 64, 3)
  # convert_to(data_sets.validation, 'validation')
  # convert_to(data_sets.test, 'test')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--directory',
      type=str,
      default='/mnt/DataBlock/CelebA/Img/img_align_celeba',
      help='Directory to download data files and write the converted result'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
