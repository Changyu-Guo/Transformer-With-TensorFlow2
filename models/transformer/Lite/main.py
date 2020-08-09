# -*- coding: utf - 8 -*-

import numpy as np
import tensorflow as tf


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) // np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.sin(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    """
    :param seq: (batch_size, seq_len)
    :return: (batch_size, 1, 1, seq_len)
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    :param size: seq_len
    :return: (size, size)
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def scaled_dot_product_attention(q, k, v, mask):
    """
    :param q: (batch_size, [num_heads], seq_len_q, hidden_size_q)
    :param k: (batch_size, [num_heads], seq_len_k, hidden_size_k)
    :param v: (batch_size, [num_heads], seq_len_v, hidden_size_v)
    :param mask: (batch_size, 1, 1, seq_len_k) or (seq_len_k, seq_len_k)
    hidden_size_q == hidden_size_k
    seq_len_k == seq_len_v
    """

    # (batch_size, [num_heads], seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        # mask 的地方无穷小
        # softmax 之后接近于 0
        scaled_attention_logits += (mask * -1e9)

    # (batch_size, [num_heads], seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # (batch_size, [num_heads], seq_len_q, seq_len_k) * (batch_size, [num_heads], seq_len_v, hidden_size_v)
    # -> (batch_size, [num_heads], seq_len_q, hidden_size_v)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = self.d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

        # (batch_size, num_heads, -1, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        # (batch_size, seq_len, d_model)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # (batch_size, num_heads, seq_len, depth)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # (batch_size, seq_len, d_model)
        output = self.dense(concat_attention)
        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, drop_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward_net = point_wise_feed_forward_network(d_model, dff)

        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout_1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout_2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        attention_output, _ = self.multi_head_attention(x, x, x, mask)
        attention_output = self.dropout_1(attention_output, training=training)
        output_1 = self.layer_norm_1(x + attention_output)

        feed_forward_net_output = self.feed_forward_net(output_1)
        feed_forward_net_output = self.dropout_2(feed_forward_net_output, training=training)
        output_2 = self.layer_norm_2(output_1 + feed_forward_net_output)

        return output_2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, drop_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.multi_head_attention_1 = MultiHeadAttention(d_model, num_heads)
        self.multi_head_attention_2 = MultiHeadAttention(d_model, num_heads)

        self.feed_forward_net = point_wise_feed_forward_network(d_model, dff)

        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout_1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout_2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout_3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        attention_1, attention_weight_block_1 = self.multi_head_attention_1(
            x, x, x, look_ahead_mask
        )
        attention_1 = self.dropout_1(attention_1, training=training)
        output_1 = self.layer_norm_1(attention_1 + x)

        attention_2, attention_weight_block_2 = self.multi_head_attention_2(
            output_1, encoder_output, encoder_output, padding_mask
        )
        attention_2 = self.dropout_2(attention_2, training=training)
        output_2 = self.layer_norm_2(attention_2 + output_1)

        feed_forward_net_output = self.feed_forward_net(output_2)
        feed_forward_net_output = self.dropout_3(feed_forward_net_output, training=training)
        output_3 = self.layer_norm_3(feed_forward_net_output + attention_2)

        return output_3, attention_weight_block_1, attention_weight_block_2


class Encoder(tf.keras.layers.Layer):
    def __init__(
            self,
            num_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            maximum_position_encoding=10000,
            drop_rate=0.1
    ):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(
            maximum_position_encoding,
            self.d_model
        )
        self.encoder_layers = [EncoderLayer(d_model, num_heads, dff, drop_rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(
            self,
            num_layers,
            d_model,
            num_heads,
            dff,
            target_vocab_size,
            maximum_position_encoding,
            drop_rate=0.1
    ):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, drop_rate)
                               for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block_1, block_2 = self.decoder_layers[i](
                x, encoder_output, training, look_ahead_mask, padding_mask
            )

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block_1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block_2

        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(
            self,
            num_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            target_vocab_size,
            pe_input,
            pe_target,
            drop_rate=0.1
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            pe_input,
            drop_rate
        )

        self.decoder = Decoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            target_vocab_size,
            pe_target,
            drop_rate
        )

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, target, training, encoder_padding_mask, look_ahead_mask, decoder_padding_mask):
        encoder_output = self.encoder(inputs, training, encoder_padding_mask)
        decoder_output, attention_weights = self.decoder(
            target, encoder_output, training, look_ahead_mask, decoder_padding_mask
        )
        final_output = self.final_layer(decoder_output)
        return final_output, attention_weights


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg_1 = tf.math.rsqrt(step)
        arg_2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg_1, arg_2)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss = tf.keras.losses.SparseCategoricalCrossentropy(real, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_mean(loss)


def create_masks(inputs, target):
    """
    :param inputs: (batch_size, seq_len)
    :param target: (batch_size, seq_len)
    """
    # (batch_size, 1, 1, seq_len)
    encoder_padding_mask = create_padding_mask(inputs)

    # (batch_size, 1, 1, seq_len)
    decoder_padding_mask = create_padding_mask(inputs)

    # (seq_len, seq_len)
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])

    # (batch_size, 1, 1, seq_len)
    decoder_target_padding_mask = create_padding_mask(target)

    # (batch_size, 1, seq_len, seq_len)
    combined_mask = tf.maximum(decoder_target_padding_mask, look_ahead_mask)
    return encoder_padding_mask, combined_mask, decoder_padding_mask


if __name__ == '__main__':
    transformer = Transformer(
        num_layers=2,
        d_model=512,
        num_heads=8,
        dff=1024,
        input_vocab_size=8216,
        target_vocab_size=8089,
        pe_input=8216,
        pe_target=8089,
        drop_rate=0.1
    )
    inputs = tf.random.uniform((32, 64))
    target = tf.random.uniform((32, 128))
    encoder_padding_mask, combined_mask, decoder_padding_mask = create_masks(inputs, target)
    transformer(inputs, target, True, encoder_padding_mask, combined_mask, decoder_padding_mask)


