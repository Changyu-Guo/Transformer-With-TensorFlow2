# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from layers import attention_layers
from layers.attention_layers import einsum_dense


class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            num_heads,
            filter_size,
            filter_activation,
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
            multi_channel_cross_attention=False,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            use_bias=True,
            norm_first=False,
            norm_epsilon=1e-12,
            **kwargs
    ):
        super(TransformerDecoderLayer, self).__init__(**kwargs)
        self._num_heads = num_heads
        self._filter_size = filter_size
        self._filter_activation = tf.keras.activations.get(
            filter_activation)
        self._dropout_rate = dropout_rate
        self._attention_dropout_rate = attention_dropout_rate
        self._multi_channel_cross_attention = multi_channel_cross_attention
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)
        self._use_bias = use_bias
        self._norm_first = norm_first
        self._norm_epsilon = norm_epsilon
        if self._multi_channel_cross_attention:
            self._cross_attention_cls = None
        else:
            self._cross_attention_cls = attention_layers.MultiHeadAttention

    def build(self, input_shape):
        target_tensor_shape = input_shape[0]
        if len(target_tensor_shape) != 3:
            raise ValueError(
                ''
            )

