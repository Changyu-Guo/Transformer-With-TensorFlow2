# -*- coding: utf - 8 -*-

import tensorflow as tf
from layers import utils


class BertPositionEmbedding(tf.keras.layers.Layer):
    """BERT"""

    def __init__(
            self,
            initializer='glorot_uniform',
            use_dynamic_slicing=False,
            max_seq_len=None,
            **kwargs
    ):
        if 'dtype' not in kwargs:
            kwargs['dtype'] = tf.float32

        super(BertPositionEmbedding, self).__init__(**kwargs)

        if use_dynamic_slicing and max_seq_len is None:
            raise ValueError(
                'If use_dynamic_slicing is True, max_seq_len must be set'
            )
        self._max_seq_len = max_seq_len
        self._initializer = tf.keras.initializers.get(initializer)
        self._use_dynamic_slicing = use_dynamic_slicing

    def get_config(self):
        config = {
            'max_seq_len': self._max_seq_len,
            'initializer': tf.keras.initializers.serialize(self._initializer),
            'use_dynamic_slicing': self._use_dynamic_slicing
        }
        base_config = super(BertPositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        dim_list = input_shape.as_list()

        if len(dim_list) != 3:
            raise ValueError(
                'BertPositionEmbedding expects a 3-dimensional input tensor '
                'of shape [batch, seq_len, hidden_size]'
            )

        seq_len = dim_list[1]
        hidden_size = dim_list[2]

        if not self._use_dynamic_slicing:
            if seq_len is None:
                raise ValueError(
                    'BertPositionEmbedding must have use_dynamic_slicing set '
                    'to True (and max_seq_len set) when the '
                    'sequence (1st) dimension of the input is None.'
                )
            if self._max_seq_len is not None:
                raise ValueError(
                    'When use_dynamic_slicing is False, max_seq_len should '
                    "not be specified and we ought to use seq_len to get the "
                    "variable shape."
                )

        if self._max_seq_len is not None:
            weight_seq_len = self._max_seq_len
        else:
            weight_seq_len = seq_len

        self.position_embeddings = self.add_weight(
            'embeddings',
            shape=[weight_seq_len, hidden_size],
            initializer=self._initializer
        )

        super(BertPositionEmbedding, self).build(input_shape)

    def call(self, inputs):
        input_shape = utils.get_shape_list(inputs, expected_rank=3)
        if self._use_dynamic_slicing:
            position_embeddings = self.position_embeddings[:input_shape[1], :]
        else:
            position_embeddings = self.position_embeddings

        return tf.broadcast_to(position_embeddings, input_shape)
