from enum import Enum
from typing import List

import tensorflow as tf
import tensorflow_addons as tfa

from attention import MultiHeadAttention


class NormType(Enum):
    BATCHNORM = 'batchnorm'
    GROUPNORM = 'groupnorm'


def normalization(norm_type: NormType, filter_num: int):
    if norm_type == NormType.BATCHNORM:
        return tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99)
    elif norm_type == NormType.GROUPNORM:
        return tfa.layers.GroupNormalization(axis=-1, groups=filter_num // 8)


# source: https://github.com/ruiqigao/recovery_likelihood/blob/main/nn.py
def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32

    half_dim = embedding_dim // 2
    emb = tf.math.log(10000.0) / float(half_dim - 1)
    emb = tf.math.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
    # emb = tf.range(num_embeddings, dtype=tf.float32)[:, None] * emb[None, :]
    emb = tf.cast(timesteps, dtype=tf.float32)[:, None] * emb[None, :]
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # zero pad
        # emb = tf.concat([emb, tf.zeros([num_embeddings, 1])], axis=1)
        emb = tf.pad(emb, [[0, 0], [0, 1]])
    assert emb.shape == [timesteps.shape[0], embedding_dim]
    return emb


class Scale(Enum):
    NONE = 1
    UP = 2
    DOWN = 3


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, scale, norm_type: NormType):
        super().__init__()

        self.filter_num = filter_num
        self.scale = scale

        self.time_embedding_dense = tf.keras.layers.Dense(filter_num, activation=None)

        self.norm1 = normalization(norm_type, filter_num)
        self.norm2 = normalization(norm_type, filter_num)
        self.conv1 = None
        self.skip_conv = None
        self.conv2 = None
        self.dropout = tf.keras.layers.Dropout(rate=0.1)

    def build(self, input_shape):
        self.skip_conv = None

        if self.scale == Scale.NONE:
            self.conv1 = tf.keras.layers.Conv2D(filters=self.filter_num, kernel_size=(3, 3), strides=(1, 1),
                                                padding="same")
            if input_shape[-1] != self.filter_num:
                self.skip_conv = tf.keras.layers.Conv2D(filters=self.filter_num, kernel_size=(1, 1), strides=(1, 1),
                                                        padding="same")
        elif self.scale == Scale.DOWN:
            self.conv1 = tf.keras.layers.Conv2D(filters=self.filter_num, kernel_size=(3, 3), strides=(2, 2),
                                                padding="same")
            self.skip_conv = tf.keras.layers.Conv2D(filters=self.filter_num, kernel_size=(3, 3), strides=(2, 2),
                                                    padding="same")
        elif self.scale == Scale.UP:
            self.conv1 = tf.keras.layers.Conv2DTranspose(filters=self.filter_num, kernel_size=(3, 3), strides=(2, 2),
                                                         padding="same")
            self.skip_conv = tf.keras.layers.Conv2DTranspose(filters=self.filter_num, kernel_size=(3, 3),
                                                             strides=(2, 2), padding="same")
        else:
            raise RuntimeError("Unknown scale policy")

        self.conv2 = tf.keras.layers.Conv2D(filters=self.filter_num, kernel_size=(3, 3), strides=(1, 1), padding="same")

    def call(self, x, time_embedding):
        if self.skip_conv is not None:
            skip = self.skip_conv(x)
        else:
            skip = x

        time_embedding = self.time_embedding_dense(time_embedding)

        x = self.norm1(x)
        x = tf.nn.swish(x)

        x = self.conv1(x)

        x += time_embedding[:, None, None, :]

        x = self.norm2(x)
        x = tf.nn.swish(x)

        x = self.dropout(x)

        x = self.conv2(x)

        return skip + x


class Network(tf.keras.Model):
    def __init__(self, num_classes, num_levels, attn_levels, norm_type: NormType, filter_num=64):
        super().__init__()

        self.num_classes = num_classes
        self.filter_num = filter_num

        self.num_levels = num_levels
        self.block_num_per_level = 2

        self.conv0 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=(1, 1), padding="same")
        self.norm0 = normalization(norm_type, filter_num)

        self.temb_dense1 = tf.keras.layers.Dense(256)
        self.temb_dense2 = tf.keras.layers.Dense(256)

        self.down_blocks: List[List[tf.keras.layers.Layer]] = [[] for _ in range(self.num_levels - 1)]
        self.down_attns: List[tf.keras.layers.Layer] = []
        for i in range(self.num_levels - 1):

            for j in range(self.block_num_per_level):
                scale = Scale.NONE if j < self.block_num_per_level - 1 else Scale.DOWN
                self.down_blocks[i].append(BasicBlock(filter_num, scale, norm_type))

            self.down_attns.append(MultiHeadAttention(d_model=filter_num, num_heads=8, attn_pdrop=0.0,
                                                      resid_pdrop=0.0) if i in attn_levels else None)

        self.mid_blocks: List[tf.keras.layers.Layer] = []
        for j in range(self.block_num_per_level):
            self.mid_blocks.append(BasicBlock(filter_num, Scale.NONE, norm_type))

        self.up_blocks: List[List[tf.keras.layers.Layer]] = [[] for _ in range(self.num_levels - 1)]
        self.up_attns: List[tf.keras.layers.Layer] = []
        for i in range(self.num_levels - 2, -1, -1):

            for j in range(self.block_num_per_level):
                scale = Scale.NONE if j > 0 else Scale.UP
                self.up_blocks[i].append(BasicBlock(filter_num, scale, norm_type))

            self.up_attns.append(MultiHeadAttention(d_model=filter_num, num_heads=8, attn_pdrop=0.0,
                                                    resid_pdrop=0.0) if i in attn_levels else None)

        self.conv_out = tf.keras.layers.Conv2D(filters=1 * self.num_classes, kernel_size=(3, 3), strides=(1, 1),
                                               padding="same", kernel_initializer=tf.keras.initializers.Zeros(),
                                               bias_initializer=tf.keras.initializers.Constant(0.0))

    @tf.function
    def call(self, x, time):

        # x = tf.cast(x, dtype=tf.float32)
        # x /= float(self.num_classes - 1) / 2.0
        # x -= 1.0

        time_embedding = get_timestep_embedding(time, self.filter_num)
        time_embedding = self.temb_dense1(time_embedding)
        time_embedding = tf.nn.swish(time_embedding)
        time_embedding = self.temb_dense2(time_embedding)
        time_embedding = tf.nn.swish(time_embedding)

        x = tf.reshape(x, [-1, 32, 32, 1*self.num_classes])
        x = self.conv0(x)

        f_list = []
        for i in range(self.num_levels - 1):
            for b in self.down_blocks[i]:
                x = b(x, time_embedding)
                f_list.append(x)
            if self.down_attns[i] is not None:
                x = self.down_attns[i](x)

        for b in self.mid_blocks:
            x = b(x, time_embedding)

        for i in range(self.num_levels - 2, -1, -1):
            if self.up_attns[i] is not None:
                x = self.up_attns[i](x)
            for b in self.up_blocks[i]:
                x = tf.concat([x, f_list.pop()], axis=-1)
                x = b(x, time_embedding)

        x = self.conv_out(x)

        x = tf.reshape(x, [-1, 32, 32, 1, self.num_classes])

        return x
