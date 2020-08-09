# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow.python.keras import keras_parameterized
from layers.transformer_layers import encoder_layer


@parameterized.named_parameters(
    ('base', encoder_layer.TransformerEncoderLayer)
)
class EncoderLayerTest(keras_parameterized.TestCase):
    def tearDown(self):
        super(EncoderLayerTest, self).tearDown()
        tf.keras.mixed_precision.experimental.set_policy('float32')

    def test_layer_creation(self, transformer_cls):
        test_layer = transformer_cls(
            num_attention_heads=10,
            intermediate_size=2048,
            intermediate_activation='relu'
        )
        seq_len = 21
        hidden_size = 80
        data_tensor = tf.keras.Input(shape=(seq_len, hidden_size))
        output_tensor = test_layer(data_tensor, training=False)
        self.assertEqual(data_tensor.shape.as_list(), output_tensor.shape.as_list())

    def test_layer_creation_with_mask(self, transformer_cls):
        test_layer = transformer_cls(
            num_attention_heads=10,
            intermediate_size=2048,
            intermediate_activation='relu'
        )
        seq_len = 21
        hidden_size = 80

        data_tensor = tf.keras.Input(shape=(seq_len, hidden_size))
        mask_tensor = tf.keras.Input(shape=(seq_len, seq_len))
        output_tensor = test_layer([data_tensor, mask_tensor], training=False)
        self.assertEqual(data_tensor.shape.as_list(), output_tensor.shape.as_list())

    def test_layer_creation_with_error_mask_fails(self, transformer_cls):
        test_layer = transformer_cls(
            num_attention_heads=10,
            intermediate_size=2048,
            intermediate_activation='relu'
        )
        seq_len = 21
        hidden_size = 80
        data_tensor = tf.keras.Input(shape=(seq_len, hidden_size))
        mask_tensor = tf.keras.Input(shape=(seq_len, seq_len - 3))
        with self.assertRaisesRegex(ValueError, 'When passing a mask tensor.*'):
            _ = test_layer([data_tensor, mask_tensor], training=True)


if __name__ == '__main__':
    tf.test.main()
