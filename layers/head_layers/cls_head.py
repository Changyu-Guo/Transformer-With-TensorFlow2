# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from layers import utils


class ClassificationHead(tf.keras.layers.Layer):
    def __init__(
            self,
            inner_dim,
            num_classes,
            cls_token_idx=0,
            activation='tanh',
            dropout_rate=0.0,
            initializer='glorot_uniform',
            **kwargs
    ):
        super(ClassificationHead, self).__init__(**kwargs)
        self._dropout_rate = dropout_rate
        self._inner_dim = inner_dim
        self._num_classes = num_classes
        self._activation = tf.keras.activations.get(activation)
        self._initializer = tf.keras.initializers.get(initializer)
        self._cls_token_idx = cls_token_idx

        self.dense = tf.keras.layers.Dense(
            units=inner_dim,
            activation=self._activation,
            kernel_initializer=self._initializer,
            name='pooler_dense'
        )
        self.dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)
        self.output_dense = tf.keras.layers.Dense(
            units=num_classes,
            kernel_initializer=self._initializer,
            name='logits'
        )

    def call(self, features):
        x = features[:, self._cls_token_idx, :]
        x = self.dense(x)
        x = self.dropout(x)
        x = self.output_dense(x)
        return x

    def get_config(self):
        config = {
            'dropout_rate': self._dropout_rate,
            'num_classes': self._num_classes,
            'inner_dim': self._inner_dim,
            'activation': tf.keras.activations.serialize(self._activation),
            'initializer': tf.keras.initializers.serialize(self._initializer)
        }
        config.update(super(ClassificationHead, self).get_config())
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def checkpoint_items(self):
        return {self.dense.name: self.dense}
