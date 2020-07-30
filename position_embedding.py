# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import utils

class RelativePositionEmbedding(tf.keras.layers.Layer):
    def __init__(
            self,
            hidden_size,
            min_timescale=1.0,
            max_timescale=1.0e4,
            **kwargs
    ):
        if 'dtype' not in kwargs:
            kwargs['dtype'] = 'float32'

        super(RelativePositionEmbedding, self).__init__(**kwargs)
        self._hidden_size = hidden_size
        self._min_timescale = min_timescale
        self._max_timescale = max_timescale

    def get_config(self):
        config = {
            'hidden_size': self._hidden_size,
            'min_timescale': self._min_timescale,
            'max_timescale': self._max_timescale
        }
        base_config = super(RelativePositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, length=None):
        if inputs is None and length is None:
            raise ValueError(
                'If inputs is None, length must be set in'
                'RelativePositionEmbedding()'
            )
        if inputs is not None:
            input_shape = utils.get_shape_list(inputs)
            if length is not None and length != input_shape[1]:
                raise ValueError(
                    'If inputs is not None, length must equal to input_shape[1]'
                )
            length = input_shape[1]

        position = tf.cast(tf.range(length), tf.float32)
        num_timescales = self._hidden_size // 2
        min_timescale, max_timescale = self._min_timescale, self._max_timescale
        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (tf.cast(num_timescales, tf.float32) - 1))
        inv_timescales = min_timescale * tf.exp(
            tf.cast(tf.range(num_timescales), tf.float32) *
            -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales,
                                                                   0)
        position_embeddings = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)],
                                        axis=1)
        return position_embeddings
