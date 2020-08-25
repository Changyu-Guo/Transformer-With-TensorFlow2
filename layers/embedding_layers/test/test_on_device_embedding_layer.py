# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from layers.embedding_layers import embedding_layer


class OnDeviceEmbeddingTest(tf.test.TestCase):
    def test_layer_creation(self):
        vocab_size = 31
        embedding_size = 27
        test_layer = embedding_layer.Embedding(
            vocab_size=vocab_size, embedding_size=embedding_size
        )
        seq_len = 23
        input_tensor = tf.keras.Input(shape=(seq_len,), dtype=tf.int32)
        output_tensor = test_layer(input_tensor)

        expected_output_shape = [None, seq_len, embedding_size]
        self.assertEqual(expected_output_shape, output_tensor.shape.as_list())
        self.assertEqual(output_tensor.dtype, tf.float32)


if __name__ == '__main__':
    tf.test.main()