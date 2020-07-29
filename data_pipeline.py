# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import logging
import tensorflow as tf
from tensorflow.python.util import nest

_READ_RECORD_BUFFER = 8 * 1000 * 1000
_MIN_BOUNDARY = 8
_BOUNDARY_SCALE = 1.1


def _load_records(filename):
    return tf.data.TFRecordDataset(filename, buffer_size=_READ_RECORD_BUFFER)


def _parse_example(serialized_example):
    data_fields = {
        'inputs': tf.io.VarLenFeature(tf.int64),
        'targets': tf.io.VarLenFeature(tf.int64)
    }
    parsed = tf.io.parse_single_example(serialized_example, data_fields)
    inputs = tf.sparse.to_dense(parsed['inputs'])
    targets = tf.sparse.to_dense(parsed['targets'])
    return inputs, targets


def _read_and_batch_from_files(
        file_pattern, batch_size, max_length, max_io_parallelism,
        shuffle, repeat, static_batch=False, num_replicas=1, ctx=None
):
    """
    :param file_pattern: 用于匹配 TFRecord 文件名的字符串
    :param batch_size: num examples per batch
    :param max_length: num tokens per example
    :param max_io_parallelism: cpu cores for parallel input processing
    :param shuffle: whether randomizes
    :param repeat: whether repeat dataset
    :param static_batch: whether batch have same shape
    :param num_replicas: num of GPUs or TPUs
    :param ctx: input context
    :return: tf.data.Dataset
    """
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)

def generate_synthetic_data(
        input_shape, input_value=0, input_dtype=None, label_shape=None,
        label_value=0, label_dtype=None
):
    element = input_element = nest.map_structure(
        lambda s: tf.constant(input_value, input_dtype, s), input_shape
    )

    if label_shape:
        label_element = nest.map_structure(
            lambda s: tf.constant(label_value, label_dtype, s), label_shape
        )
        element = (input_element, label_element)
    return tf.data.Dataset.from_tensors(element).repeat()


def _generate_synthetic_data(params):
    batch_size = int(params['batch_size'] // params['max_length'])
    length = params['max_length']
    dataset = generate_synthetic_data(
        input_shape=tf.TensorShape([length]),
        input_value=1,
        input_dtype=tf.int64,
        label_shape=tf.TensorShape([length]),
        label_value=-1,
        label_dtype=tf.int64
    )
    if params['static_batch']:
        dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
        dataset = dataset.padded_batch(batch_size, ([None], [None]))
    return dataset


def train_input_fn(params, ctx=None):
    file_pattern = os.path.join(params['data_dir'] or '', '*train*')
    if params['use_synthetic_data']:
        return _generate_synthetic_data(params)
    return _read_and_batch_from_files(
        file_pattern, params['batch_size'], params['max_length'],
        params['max_io_parallelism'], shuffle=True, repeat=params['repeat_dataset'],
        static_batch=params['static_batch'], num_replicas=params['num_gpus'], ctx=ctx
    )
