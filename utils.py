import numpy as np
import tensorflow as tf


def random_times(n, max_time):
    return np.random.randint(1, max_time, size=[n])


def sample_categorical_images(logits):
    num_classes = logits.shape[-1]
    c = tf.random.categorical(tf.reshape(logits, [-1, num_classes]), 1, dtype=tf.int32)
    return tf.reshape(c, logits.shape[:-1])


def one_hot_images(x, num_classes, dtype):
    H, W, C = x.shape[1:4]
    indices = tf.reshape(x, [-1, H * W * C])
    out = tf.one_hot(indices, depth=num_classes, dtype=dtype)
    return tf.reshape(out, [-1, H, W, C, num_classes])
