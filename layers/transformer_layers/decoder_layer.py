# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from layers.attention_layers.multi_head_attention_layer import CacheAttention


class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            num_attention_heads,
            intermediate_size,
            intermediate_activation,
            hidden_dropout_rate=0.0,
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
            intermediate_dropout_rate=0.0,
            attention_initializer=None,
            **kwargs
    ):
        super(TransformerDecoderLayer, self).__init__(**kwargs)
        self._num_attention_heads = num_attention_heads
        self._intermediate_size = intermediate_size
        self._intermediate_activation = tf.keras.activations.get(
            intermediate_activation)
        self._hidden_dropout_rate = hidden_dropout_rate
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
        self._intermediate_dropout_rate = intermediate_dropout_rate
        if attention_initializer:
            self._attention_initializer = tf.keras.initializers.get(
                attention_initializer
            )
        else:
            self._attention_initializer = self._kernel_initializer
        if self._multi_channel_cross_attention:
            self._cross_attention_cls = None
        else:
            self._cross_attention_cls = attention_layers.MultiHeadAttention

    def build(self, input_shape):
        target_tensor_shape = tf.TensorShape(input_shape[0])
        if len(target_tensor_shape) != 3:
            raise ValueError(
                'need 3-dim input'
            )
        hidden_size = target_tensor_shape[2]
        if hidden_size % self._num_attention_heads != 0:
            raise ValueError(
                'division'
            )

        self.size_per_head_for_query_and_key = int(hidden_size / self._num_attention_heads)
        common_kwargs = dict(
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint
        )

        self.self_attention = CacheAttention(
            num_attention_heads=self._num_attention_heads,
            size_per_head_for_query_and_key=self.size_per_head_for_query_and_key,
            attention_dropout_rate=self._attention_dropout_rate,
            use_bias=self._use_bias,
            kernel_initializer=self._attention_initializer,
            name='self_attention',
            **common_kwargs
        )

    def call(self, inputs, cache=None, decode_loop_step=None):
        (
            input_tensor,
            encoder_output,
            encoder_decoder_attention_mask,
            self_attention_mask
        ) = inputs[:4]
        source_tensor = input_tensor
        if self_norm_first:
            pass
        self_attention_output, cache = self.self_attention(
            query=input_tensor,
            value=input_tensor,
            cache=cache,
            decode_loop_step=decode_loop_step
        )
