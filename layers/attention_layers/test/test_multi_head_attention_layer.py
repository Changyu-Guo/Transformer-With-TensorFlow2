# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow.python.keras import keras_parameterized
from layers.attention_layers import multi_head_attention_layer


class MultiHeadAttentionTest(keras_parameterized.TestCase):
    @parameterized.named_parameters(
        (
            'key_value_same_hidden_size',
            None,
            None,
            [40, 80]
        ),
        (
            'key_value_different_hidden_size',
            32,
            60,
            [40, 60]
        )
    )
    def test_non_masked_attention(
            self,
            value_size,
            output_shape,
            output_dims
    ):
        test_layer = multi_head_attention_layer.MultiHeadAttention(
            num_attention_heads=12,
            size_per_head_for_query_and_key=64,
            size_per_head_for_value=value_size,
            output_shape=output_shape
        )

        # (batch_size=None, seq_len=40, hidden_size=80)
        query = tf.keras.Input(shape=(40, 80))

        # (batch_size=None, seq_len=20, hidden_size=80)
        value = tf.keras.Input(shape=(20, 80))

        output = test_layer(query=query, value=value)
        self.assertEqual(output.shape.as_list(), [None] + output_dims)

    def test_non_masked_self_attention(self):
        test_layer = multi_head_attention_layer.MultiHeadAttention(
            num_attention_heads=12,
            size_per_head_for_query_and_key=64
        )
        query = tf.keras.Input(shape=(40, 80))
        output = test_layer(query, query)
        self.assertEqual(output.shape.as_list(), [None, 40, 80])

    def test_attention_scores(self):
        test_layer = multi_head_attention_layer.MultiHeadAttention(
            num_attention_heads=12,
            size_per_head_for_query_and_key=64,
            return_attention_scores=True
        )
        query = tf.keras.Input(shape=(40, 80))
        output, scores = test_layer(query, query)
        self.assertEqual(output.shape.as_list(), [None, 40, 80])
        # (batch_size, num_heads, seq_len, seq_len)
        self.assertEqual(scores.shape.as_list(), [None, 12, 40, 40])

    @parameterized.named_parameters(
        ('with_bias', True),
        ('no_bias', False)
    )
    def test_masked_attention(self, use_bias):
        test_layer = multi_head_attention_layer.MultiHeadAttention(
            num_attention_heads=2,
            size_per_head_for_query_and_key=2,
            use_bias=use_bias
        )
        batch_size = 3
        query = tf.keras.Input(shape=(4, 8))
        value = tf.keras.Input(shape=(2, 8))
        mask_tensor = tf.keras.Input(shape=(4, 2))
        output = test_layer(
            query=query,
            value=value,
            attention_mask=mask_tensor
        )
        model = tf.keras.Model([query, value, mask_tensor], output)

        inputs = 10 * np.random.random_sample((batch_size, 4, 8))
        targets = 10 * np.random.random_sample((batch_size, 2, 8))

        mask_data = np.random.randint(2, size=(batch_size, 4, 2))
        masked_output_data = model.predict([inputs, targets, mask_data])

        null_mask_data = np.ones((batch_size, 4, 2))
        unmasked_output_data = model.predict([inputs, targets, null_mask_data])

        self.assertNotAllClose(masked_output_data, unmasked_output_data)

        key = tf.keras.Input(shape=(2, 8))
        output = test_layer(
            query=query,
            value=value,
            key=key,
            attention_mask=mask_tensor
        )
        model = tf.keras.Model([query, value, key, mask_tensor], output)
        masked_output_data = model.predict([inputs, targets, targets, mask_data])
        unmasked_output_data = model.predict([inputs, targets, targets, null_mask_data])

        self.assertNotAllClose(masked_output_data, unmasked_output_data)

        if use_bias:
            self.assertLen(test_layer._query_dense.trainable_variables, 2)
            self.assertLen(test_layer._output_dense.trainable_variables, 2)
        else:
            self.assertLen(test_layer._query_dense.trainable_variables, 1)
            self.assertLen(test_layer._output_dense.trainable_variables, 1)

    def test_initializer(self):
        test_layer = multi_head_attention_layer.MultiHeadAttention(
            num_attention_heads=12,
            size_per_head_for_query_and_key=64,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02)
        )
        query = tf.keras.Input(shape=(40, 80))
        output = test_layer(query, query)
        self.assertEqual(output.shape.as_list(), [None, 40, 80])

    @parameterized.named_parameters(
        ('4D_inputs_one_free_batch', [3, 4], [3, 2], [4, 2], (2,)),
        ('4D_inputs_2D_attention', [3, 4], [3, 2], [3, 4, 3, 2], (1, 2)),
        ('5D_inputs_2D_attention', [5, 3, 4], [5, 3, 2], [3, 4, 3, 2], (2, 3))
    )
    def test_high_dim_attention(self, q_dims, v_dims, mask_dims, attention_axes):
        test_layer = multi_head_attention_layer.MultiHeadAttention(
            num_attention_heads=2,
            size_per_head_for_query_and_key=2,
            attention_axes=attention_axes
        )
        batch_size, hidden_size = 3, 8
        # (3, 3, 4, 8)
        query_shape = [batch_size] + q_dims + [hidden_size]
        # (3, 3, 2, 8)
        value_shape = [batch_size] + v_dims + [hidden_size]
        # (3, 3, 4, 3, 2)
        mask_shape = [batch_size] + mask_dims

        query = 10 * np.random.random_sample(query_shape)
        value = 10 * np.random.random_sample(value_shape)

        mask_data = np.random.randint(2, size=mask_shape).astype('bool')
        output = test_layer(query=query, value=value, attention_mask=mask_data)

        null_mask_data = np.ones(mask_shape)
        unmasked_output = test_layer(
            query=query,
            value=value,
            attention_mask=null_mask_data
        )
        self.assertNotAllClose(output, unmasked_output)


if __name__ == '__main__':
    tf.test.main()
