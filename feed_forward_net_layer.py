# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, hidden_size, output_size, drop_rate):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.drop_rate = drop_rate

    def build(self, input_shape):
        self.hidden_dense_layer = tf.keras.layers.Dense(
            self.hidden_size,
            use_bias=True,
            activation=tf.nn.relu,
            name='hidden_layer'
        )
        self.output_dense_layer = tf.keras.layers.Dense(
            self.output_size,
            use_bias=True,
            name='output_layer'
        )
        super(FeedForwardNetwork, self).build(input_shape)

    def get_config(self):
        return {
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'drop_rate': self.drop_rate
        }

    def call(self, x, training):
        hidden = self.hidden_dense_layer(x)
        if training:
            hidden = tf.nn.dropout(hidden, rate=self.drop_rate)
        output = self.output_dense_layer(hidden)
        return output
