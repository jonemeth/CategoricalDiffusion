from enum import Enum

import numpy as np
import tensorflow as tf

from network import Network
from utils import sample_categorical_images, one_hot_images


class CategoricalDiffusion:
    class Schedule(Enum):
        LINEAR = 'linear'
        EXPONENTIAL = 'exponential'

    def __init__(self, image_shape, num_classes: int, args):
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.time_steps = args.time_steps
        self.schedule = args.schedule

        self.network = Network(self.image_shape, num_classes, num_levels=4, attn_levels=[0, 1, 2], norm_type=args.norm_type)

    def compute_loss(self, x0, time):
        prior_prob = 1.0 / self.num_classes
        time_ = tf.reshape(time, [-1, 1, 1, 1, 1])
        x0 = one_hot_images(x0, self.num_classes, dtype=tf.float32)

        if self.schedule == CategoricalDiffusion.Schedule.LINEAR:
            alpha_hat_t = 1.0 - 1.0 * (time_ / self.time_steps)
            alpha_hat_tm1 = 1.0 - 1.0 * ((time_ - 1) / self.time_steps)
            alpha = alpha_hat_t / alpha_hat_tm1
        elif self.schedule == CategoricalDiffusion.Schedule.EXPONENTIAL:
            alpha = 0.001 ** (1.0 / self.time_steps)
            alpha_hat_t = alpha ** time_
            alpha_hat_tm1 = alpha ** (time_ - 1)
        else:
            raise RuntimeError

        xt_probs = alpha_hat_t * x0 + (1.0 - alpha_hat_t) * prior_prob
        xt_logits = tf.math.log(xt_probs)

        xt = sample_categorical_images(xt_logits)
        xt = one_hot_images(xt, self.num_classes, dtype=tf.float32)

        q_xtm1_xt_logits = self.network(xt, time, training=True)

        q_xtm1_xt = tf.nn.softmax(q_xtm1_xt_logits, axis=-1)

        p_xt_xtm1 = (alpha * xt + (1.0 - alpha) * prior_prob)
        p_xtm1_x0 = (alpha_hat_tm1 * x0 + (1.0 - alpha_hat_tm1) * prior_prob)
        p_xt_x0 = tf.reduce_sum(xt * (alpha_hat_t * x0 + (1.0 - alpha_hat_t) * prior_prob), axis=4, keepdims=True)

        p_xtm1_xtx0 = p_xt_xtm1 * p_xtm1_x0 / p_xt_x0

        KL = tf.reduce_sum(p_xtm1_xtx0 * (tf.math.log(1e-12 + p_xtm1_xtx0) - tf.math.log(1e-12 + q_xtm1_xt)), axis=4)

        loss = tf.reduce_sum(KL, axis=[1, 2, 3])
        loss = tf.reduce_mean(loss)

        return loss

    def compute_apply_gradients(self, optimizer, x, time):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x, time)

        gradients = tape.gradient(loss, self.network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

        return loss

    def generate_images(self, n):
        logits = np.zeros(shape=[n]+self.image_shape+[self.num_classes], dtype=np.float32)

        for t in range(self.time_steps, 0, -1):
            times = [t] * n
            times = tf.convert_to_tensor(times, dtype=tf.float32)
            samples = sample_categorical_images(logits)
            samples = one_hot_images(samples, self.num_classes, dtype=tf.float32)

            logits = self.network(samples, times, training=False)

        return sample_categorical_images(logits)
