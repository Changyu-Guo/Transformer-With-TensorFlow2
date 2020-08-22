# -*- coding: utf - 8 -*-

import math
from datetime import datetime
import tensorflow as tf
from absl import logging

logging.set_verbosity(logging.INFO)


def loss_fn(targets_ids, output):
    cross_entropy = lambda x: tf.keras.losses.sparse_categorical_crossentropy(x[0], x[1], from_logits=True)
    return cross_entropy((targets_ids, output))


def create_padding_mask(seqs):
    return tf.cast(tf.equal(seqs, 0)[:, None, None, :], tf.float32)


def create_look_ahead_mask(seq_len):
    mask = 1 - tf.linalg.band_part(tf.ones(shape=(seq_len, seq_len)), num_lower=-1, num_upper=0)
    return mask[None, :, :]


class Network(tf.keras.Model):
    def __init__(self, name, num_heads, hidden_size, vocab_size, seq_len):
        super(Network, self).__init__(name=name)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def build(self, input_shape):
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.hidden_size)

        self.query_dense = tf.keras.layers.Dense(units=self.hidden_size)
        self.key_dense = tf.keras.layers.Dense(units=self.hidden_size)
        self.value_dense = tf.keras.layers.Dense(units=self.hidden_size)

        self.output_dense = tf.keras.layers.Dense(units=self.vocab_size)

        super(Network, self).build(input_shape)

    def call(self, inputs):
        print(inputs)
        inputs = inputs[0]
        inputs_shape = tf.shape(inputs)
        batch_size, seq_len = inputs_shape[0], inputs_shape[1]

        embedded_inputs = self.embedding(inputs)

        query = self.query_dense(embedded_inputs)
        key = self.key_dense(embedded_inputs)
        value = self.value_dense(embedded_inputs)

        # split heads
        size_per_head = int(self.hidden_size // self.num_heads)

        query = tf.reshape(query, shape=(batch_size, seq_len, self.num_heads, size_per_head))
        query = tf.transpose(query, [0, 2, 1, 3])

        key = tf.reshape(key, shape=(batch_size, seq_len, self.num_heads, size_per_head))
        key = tf.transpose(key, [0, 2, 1, 3])

        value = tf.reshape(value, shape=(batch_size, seq_len, self.num_heads, size_per_head))
        value = tf.transpose(value, [0, 2, 1, 3])

        # calc weights
        attention_weights = tf.matmul(query, key, transpose_b=True)
        attention_weights = tf.multiply(attention_weights, 1.0 / math.sqrt(size_per_head))

        # masked softmax
        self_attention_mask = create_padding_mask(inputs)
        attention_weights = tf.math.log_softmax(attention_weights + (1 - self_attention_mask * -1e-9))

        # attention
        attention_output = tf.matmul(attention_weights, value)
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, shape=(batch_size, seq_len, self.hidden_size))

        # calc output
        output = self.output_dense(attention_output)

        return output


def create_toy_dataset(batch_size, seq_len, vocab_size):
    def gen():
        for _ in range(10):
            x = tf.random.uniform(minval=0, maxval=vocab_size, shape=(seq_len,), dtype=tf.int32)
            y = tf.random.uniform(minval=0, maxval=vocab_size, shape=(seq_len,), dtype=tf.int32)
            z = tf.random.uniform(minval=0, maxval=vocab_size, shape=(seq_len,), dtype=tf.int32)
            yield {
                'inputs_ids': x,
                'targets_ids': y
            }, z

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_types=({'inputs_ids': tf.int32, 'targets_ids': tf.int32}, tf.int32),
    ).repeat().batch(batch_size)
    return dataset


def create_model(num_heads, hidden_size, vocab_size, seq_len):
    inputs_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name='inputs_ids')
    targets_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name='targets_ids')
    internal_model = Network(
        name='simple_attention',
        num_heads=num_heads,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        seq_len=seq_len
    )
    output = internal_model([inputs_ids, targets_ids])
    model = tf.keras.Model({
        'inputs_ids': inputs_ids,
        'targets_ids': targets_ids
    }, output)
    return model


def main():
    batch_size = 2
    num_heads = 4
    hidden_size = 512
    vocab_size = 11
    seq_len = 12
    log_dir = './logs' + datetime.now().strftime('%Y%m%d-%H%M%S')

    dataset = create_toy_dataset(batch_size, seq_len, vocab_size)

    model = create_model(num_heads, hidden_size, vocab_size, seq_len)
    optimizer = tf.keras.optimizers.Adam(0.1)
    model.compile(
        optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    )
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
    model.fit(
        x=dataset,
        y=None,
        epochs=1,
        steps_per_epoch=1,
        # callbacks=[tensorboard_callback]
    )


if __name__ == '__main__':
    main()
