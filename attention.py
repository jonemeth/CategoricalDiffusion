import tensorflow as tf

# source: https://github.com/kamalkraj/minGPT-TF/blob/master/mingpt/model.py


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """
    def __init__(self, d_model, num_heads, attn_pdrop, resid_pdrop):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        # key, query, value projections for all heads
        self.wq = tf.keras.layers.Dense(d_model,
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                                        name="query")
        self.wk = tf.keras.layers.Dense(d_model,
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                                        name="key")
        self.wv = tf.keras.layers.Dense(d_model,
                                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02),
                                        name="value")
        # regularization
        self.attn_drop = tf.keras.layers.Dropout(rate=attn_pdrop)
        self.resid_drop = tf.keras.layers.Dropout(rate=resid_pdrop)
        # output projection
        self.dense = tf.keras.layers.Dense(d_model, name="projection")

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, training, mask=None):
        input_shape = list(x.shape[1:])

        batch_size = tf.shape(x)[0]

        x = tf.reshape(x, [batch_size, input_shape[0] * input_shape[1], self.d_model])

        q = self.wq(x)  # (batch_size, seq_len, d_model)
        k = self.wk(x)  # (batch_size, seq_len, d_model)
        v = self.wv(x)  # (batch_size, seq_len, d_model)
        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # (..., seq_len_q, seq_len_k)
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(
            scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        attention_weights = self.attn_drop(
            attention_weights, training=training)
        # (..., seq_len_q, depth_v)
        scaled_attention = tf.matmul(attention_weights, v)
        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)
        output = self.resid_drop(output, training=training)

        output = tf.reshape(output, [-1] + input_shape)

        return output
