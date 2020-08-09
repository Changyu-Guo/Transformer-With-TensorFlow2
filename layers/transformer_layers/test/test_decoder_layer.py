# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from layers.transformer_layers.decoder_layer import TransformerDecoderLayer


def _create_cache(batch_size, init_decode_length, num_heads, head_size):
    return {
        'key': tf.zeros(
            [batch_size, init_decode_length, num_heads, head_size],
            dtype=tf.float32
        ),
        'value': tf.zeros(
            [batch_size, init_decode_length, num_heads, head_size],
            dtype=tf.float32
        )
    }


class TransformerDecoderLayerTest(tf.test.TestCase):

    def test_decoder_block_with_cache(self):
        num_attention_heads = 2
        hidden_size = 16
        decoder_block = TransformerDecoderLayer(
            num_attention_heads=num_attention_heads,
            intermediate_size=32,
            intermediate_activation='relu',
            hidden_dropout_rate=0.1,
            attention_dropout_rate=0.1
        )
        inputs_tensor = tf.zeros([2, 4, 16], dtype=tf.float32)
        inputs_mask = tf.zeros([2, 4, 4], dtype=tf.float32)
        inputs = [inputs_tensor, inputs_tensor, inputs_mask, inputs_mask]
        cache = _create_cache(2, 0, num_attention_heads, hidden_size // num_attention_heads)
        output, cache = decoder_block(inputs=inputs, training=False, cache=cache)
        self.assertEqual(output.shape, (2, 4, hidden_size))
        self.assertEqual(cache['value'].shape, (2, 4, 2, 8))

    def test_use_bias_norm_first(self):
        num_attention_heads = 2
        hidden_size = 16
        decoder_block = TransformerDecoderLayer(
            num_attention_heads=num_attention_heads,
            intermediate_size=32,
            intermediate_activation='relu',
            hidden_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            use_bias=False,
            norm_first=True,
            norm_epsilon=1e-6
        )

        targets_tensor = tf.zeros([2, 4, 16], dtype=tf.float32)
        targets_mask = tf.zeros([2, 4, 4], dtype=tf.float32)
        inputs = [targets_tensor, targets_tensor, targets_mask, targets_mask]
        output, _ = decoder_block(inputs, training=True)
        self.assertEqual(output.shape, (2, 4, hidden_size))


if __name__ == '__main__':
    tf.test.main()