# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

def get_shape_list(tensor, expected_rank=None, name=None):
    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def assert_rank(tensor, expected_rank, name=None):
    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.dims
    if actual_rank not in expected_rank_dict:
        raise ValueError(
            "For the tensor `%s`, the actual tensor rank `%d` (shape = %s) is not "
            "equal to the expected tensor rank `%s`" %
            (name, actual_rank, str(tensor.shape), str(expected_rank))
        )


def get_padding(x, padding_value=0, dtype=tf.float32):
    with tf.name_scope('padding'):
        return tf.cast(tf.equal(x, padding_value), dtype)


def get_padding_bias(x, padding_value=0, dtype=tf.float32):
    with tf.name_scope('attention_bias'):
        padding = get_padding(x, padding_value, dtype)
        attention_bias = padding * _NEG_INF_FP32
        attention_bias = tf.expand_dims(
            tf.expand_dims(attention_bias, axis=1), axis=1
        )
    return attention_bias
