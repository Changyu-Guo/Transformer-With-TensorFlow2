# -*- coding: utf - 8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from layers import utils


class SelfAttentionMask(tf.keras.layers.Layer):
    def call(self, inputs):

        # (batch_size, seq_len_from, hidden_size)
        from_tensor = inputs[0]

        # (batch_size, seq_len_from, seq_len_to)
        to_mask = inputs[1]

        from_shape = utils.get_shape_list(from_tensor, expected_rank=[2, 3])
        batch_size = from_shape[0]
        from_seq_len = from_shape[1]

        to_shape = utils.get_shape_list(to_mask, expected_rank=2)
        to_seq_len = to_shape[1]

        to_mask = tf.cast(
            tf.reshape(to_mask, [batch_size, 1, to_seq_len]),
            dtype=from_tensor.dtype
        )
        broadcast_ones = tf.ones(
            shape=[batch_size, from_seq_len, 1],
            dtype=from_tensor.dtype
        )
        mask = broadcast_ones * to_mask

        return mask