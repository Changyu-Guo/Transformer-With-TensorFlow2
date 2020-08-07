# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def decode_record(record, name_to_features):
    example = tf.io.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t

    return example


def create_pretrain_dataset(
        input_patterns,
        seq_len,
        max_predictions_per_seq,
        batch_size,
        is_training=True,
        input_pipeline_context=None,
        use_next_sentence_label=True,
        use_position_id=False,
        output_fake_labels=True
):
    name_to_features = {
        'input_ids': tf.io.FixedLenFeature([seq_len], tf.int64),
        'input_mask': tf.io.FixedLenFeature([seq_len], tf.int64),
        'segment_ids': tf.io.FixedLenFeature([seq_len], tf.int64),
        'masked_lm_positions': tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        'masked_lm_ids': tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        'masked_lm_weights': tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32)
    }
    if use_next_sentence_label:
        name_to_features['next_sentence_labels'] = tf.io.FixedLenFeature([1], tf.int64)

    if use_position_id:
        name_to_features['position_ids'] = tf.io.FixedLenFeature([seq_len], tf.int64)

    for input_pattern in input_patterns:
        if not tf.io.gfile.glob(input_pattern):
            raise ValueError(
                '%s dose not match any files.' % input_pattern
            )

    dataset = tf.data.Dataset.list_files(input_patterns, shuffle=is_training)

    if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
        dataset = dataset.shard(
            input_pipeline_context.num_input_pipelines,
            input_pipeline_context.input_pipeline_id
        )

    if is_training:
        dataset = dataset.repeat()

        input_files = []
        for input_pattern in input_patterns:
            input_files.extend(tf.io.gfile.glob(input_pattern))
        dataset = dataset.shuffle(len(input_files))

    dataset = dataset.interleave(
        tf.data.TFRecordDataset,
        cycle_length=8,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    if is_training:
        dataset = dataset.shuffle(100)

    decode_fn = lambda record: decode_record(record, name_to_features)
    dataset = dataset.map(decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def _select_data_from_record(record):
        x = {
            'input_word_ids': record['input_ids'],
            'input_mask': record['input_mask'],
            'input_type_ids': record['segment_ids'],
            'masked_lm_positions': record['masked_lm_positions'],
            'masked_lm_ids': record['masked_lm_ids'],
            'masked_lm_weights': record['masked_lm_weights']
        }
        if use_next_sentence_label:
            x['next_sentence_labels'] = record['next_sentence_labels']
        if use_position_id:
            x['position_ids'] = record['position_ids']

        if output_fake_labels:
            return x, record['masked_lm_weights']
        else:
            return x

    dataset = dataset.map(
        _select_data_from_record,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
