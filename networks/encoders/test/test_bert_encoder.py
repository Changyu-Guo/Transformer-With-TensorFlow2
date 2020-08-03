# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow.python.keras import keras_parameterized
from networks.encoders import bert_encoder


class BertEncoderTest(keras_parameterized.TestCase):
    def tearDown(self):
        super(BertEncoderTest, self).tearDown()
        tf.keras.mixed_precision.experimental.set_policy('float32')

    def test_network_creation(self):
        hidden_size = 32
        seq_len = 21
        test_network = bert_encoder.BertEncoder(
            vocab_size=100,
            hidden_size=hidden_size,
            num_attention_heads=2,
            num_layers=3
        )

        words_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32)
        mask = tf.keras.Input(shape=(seq_len,), dtype=tf.int32)
        type_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32)
        data, pooled = test_network([words_ids, mask, type_ids])

        self.assertIsInstance(test_network.transformer_layers, list)
        self.assertLen(test_network.transformer_layers, 3)
        self.assertIsInstance(test_network.pooler_layer, tf.keras.layers.Dense)

        expected_data_shape = [None, seq_len, hidden_size]
        expected_pooled_shape = [None, hidden_size]
        self.assertAllEqual(expected_data_shape, data.shape.as_list())
        self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

        self.assertAllEqual(tf.float32, data.dtype)
        self.assertAllEqual(tf.float32, pooled.dtype)

    def test_all_encoder_outputs_network_creation(self):
        hidden_size = 32
        seq_len = 21

        test_network = bert_encoder.BertEncoder(
            vocab_size=100,
            hidden_size=hidden_size,
            num_attention_heads=2,
            num_layers=3,
            return_all_encoder_outputs=True)

        word_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32)
        mask = tf.keras.Input(shape=(seq_len,), dtype=tf.int32)
        type_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32)
        all_encoder_outputs, pooled = test_network([word_ids, mask, type_ids])

        expected_data_shape = [None, seq_len, hidden_size]
        expected_pooled_shape = [None, hidden_size]
        self.assertLen(all_encoder_outputs, 3)
        for data in all_encoder_outputs:
            self.assertAllEqual(expected_data_shape, data.shape.as_list())
        self.assertAllEqual(expected_pooled_shape, pooled.shape.as_list())

        self.assertAllEqual(tf.float32, all_encoder_outputs[-1].dtype)
        self.assertAllEqual(tf.float32, pooled.dtype)

    @parameterized.named_parameters(
        ("all_sequence", None, 21),
        ("output_range", 1, 1),
    )
    def test_network_invocation(self, output_range, out_seq_len):
        hidden_size = 32
        seq_len = 21
        vocab_size = 57
        num_types = 7
        # Create a small TransformerEncoder for testing.
        test_network = bert_encoder.BertEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=2,
            num_layers=3,
            type_vocab_size=num_types,
            output_range=output_range)
        self.assertTrue(
            test_network._position_embedding_layer.use_dynamic_slicing)
        # Create the inputs (note that the first dimension is implicit).
        word_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32)
        mask = tf.keras.Input(shape=(seq_len,), dtype=tf.int32)
        type_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32)
        data, pooled = test_network([word_ids, mask, type_ids])

        # Create a model based off of this network:
        model = tf.keras.Model([word_ids, mask, type_ids], [data, pooled])

        # Invoke the model. We can't validate the output data here (the model is too
        # complex) but this will catch structural runtime errors.
        batch_size = 3
        word_id_data = np.random.randint(
            vocab_size, size=(batch_size, seq_len))
        mask_data = np.random.randint(2, size=(batch_size, seq_len))
        type_id_data = np.random.randint(
            num_types, size=(batch_size, seq_len))
        _ = model.predict([word_id_data, mask_data, type_id_data])

        # Creates a TransformerEncoder with max_seq_len != sequence_length
        max_seq_len = 128
        test_network = bert_encoder.BertEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            num_attention_heads=2,
            num_layers=3,
            type_vocab_size=num_types)
        self.assertTrue(test_network._position_embedding_layer.use_dynamic_slicing)
        model = tf.keras.Model([word_ids, mask, type_ids], [data, pooled])
        outputs = model.predict([word_id_data, mask_data, type_id_data])
        self.assertEqual(outputs[0].shape[1], out_seq_len)

        # Creates a TransformerEncoder with embedding_width != hidden_size
        test_network = bert_encoder.BertEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            num_attention_heads=2,
            num_layers=3,
            type_vocab_size=num_types,
            embedding_size=16)
        model = tf.keras.Model([word_ids, mask, type_ids], [data, pooled])
        outputs = model.predict([word_id_data, mask_data, type_id_data])
        self.assertEqual(outputs[0].shape[-1], hidden_size)
        self.assertTrue(hasattr(test_network, "_embedding_projection"))


if __name__ == '__main__':
    tf.test.main()
