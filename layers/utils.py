# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import numpy as np
import tensorflow as tf
from activations import gelu, swish


def get_activation(identifier):
    if isinstance(identifier, six.string_types):
        name_to_fn = {
            'gelu': gelu.gelu,
            'simple_swish': swish.simple_swish,
            'hard_swish': swish.hard_swish,
            'identity': swish.identity
        }
        identifier = str(identifier).lower()
        if identifier in name_to_fn:
            return tf.keras.activations.get(name_to_fn[identifier])
    return tf.keras.activations.get(identifier)


def get_shape_list(tensor, expected_rank=None, name=None):
    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
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

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        raise ValueError(
            "For the tensor `%s`, the actual tensor rank `%d` (shape = %s) is not "
            "equal to the expected tensor rank `%s`" %
            (name, actual_rank, str(tensor.shape), str(expected_rank))
        )


def get_padding_mask(seqs, padding_value=0, dtype=tf.float32):
    """
    :param seqs: (batch_size, seq_len)
    :param padding_value: 0
    :param dtype:
    :return: (batch_size, 1, seq_len)
    """
    with tf.name_scope('padding_mask'):
        padding_mask = tf.cast(tf.equal(seqs, padding_value), dtype)
        padding_mask = tf.expand_dims(padding_mask, axis=1)
    return padding_mask


def get_look_ahead_mask(length, dtype=tf.float32):
    """
    :param length: seq_len
    :param dtype:
    :return: (seq_len, seq_len)
    """
    with tf.name_scope('look_ahead_mask'):
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones([length, length], dtype=dtype), -1, 0)
        return look_ahead_mask[tf.newaxis, :, :]


def get_combine_mask(seqs, padding_value=0, dtype=tf.float32):
    return tf.maximum(get_padding_mask(seqs, padding_value, dtype), get_look_ahead_mask(tf.shape(seqs)[1]))
