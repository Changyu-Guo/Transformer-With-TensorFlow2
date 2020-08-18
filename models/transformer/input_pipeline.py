# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from absl import logging


def _read_and_batch_from_files(
        file_pattern,
        batch_size,
        max_seq_len,
        shuffle,
        repeat
):
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)

    dataset = dataset.interleave(
        lambda filename: tf.data.TFRecordDataset(
            filename,
            compression_type=None,
            buffer_size=None,
            num_parallel_reads=tf.data.experimental.AUTOTUNE
        ),
        cycle_length=8,
        block_length=1,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False
    )

    def _parse_example(example):
        data_fields = {
            'inputs': tf.io.VarLenFeature(tf.int64),
            'targets': tf.io.VarLenFeature(tf.int64)
        }
        parsed = tf.io.parse_single_example(example, data_fields)

        inputs = tf.sparse.to_dense(parsed['inputs'])
        targets = tf.sparse.to_dense(parsed['targets'])

        return inputs, targets

    dataset = dataset.map(
        _parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    def _filter_max_len(example, max_len=256):
        return tf.logical_and(
            tf.size(example[0]) <= max_len,
            tf.size(example[1]) <= max_len
        )

    dataset = dataset.filter(lambda x, y: _filter_max_len((x, y), max_seq_len))

    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=([max_seq_len], [max_seq_len]),
        padding_values=([0], [0]),
        drop_remainder=True
    )
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def get_train_dataset(params):
    file_pattern = os.path.join(params['data_dir'] or '', '*train*')
    return _read_and_batch_from_files(
        file_pattern,
        params['batch_size'],
        params['max_seq_len'],
        shuffle=True,
        repeat=params['repeat_dataset']
    )


def get_eval_dataset(params):
    file_pattern = os.path.join(params['data_dir'] or '', '*dev*')
    return _read_and_batch_from_files(
        file_pattern,
        params['batch_size'],
        params['max_seq_len'],
        shuffle=False,
        repeat=False
    )