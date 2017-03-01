import tensorflow as tf


def create_celeba_pipeline(filename, width=64, height=64, depth=3, batch_size=64, name=None):
    with tf.variable_scope(name):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'depth': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
            })

        image = tf.decode_raw(features['image_raw'], tf.float32)
        image.set_shape(height * width * depth)
        image = tf.reshape(image, (height, width, depth))

        pipeline = tf.train.shuffle_batch(
            [image], batch_size=batch_size, num_threads=4,
            capacity=1000 + 3 * batch_size, min_after_dequeue=1000)

        return pipeline
