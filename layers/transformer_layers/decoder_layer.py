# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from layers.attention_layers.multi_head_attention_layer import MultiHeadAttention
from layers.attention_layers.multi_head_attention_layer import CacheAttention
from layers.attention_layers.einsum_dense import EinsumDense


class TransformerDecoderLayer(tf.keras.layers.Layer):
    """
        Transformer 的解码器

        主要包括如下结构：
            1. self attention layer
            2. encoder decoder attention layer / cross attention layer
            3. position-wise feed-forward network
    """
    def __init__(
            self,
            num_attention_heads,
            intermediate_size,
            intermediate_activation,
            hidden_dropout_rate=0.0,
            attention_dropout_rate=0.0,
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

        self._num_attention_heads = num_attention_heads
        self._attention_dropout_rate = attention_dropout_rate
        self._norm_first = norm_first
        self._use_bias = use_bias
        self._norm_epsilon = norm_epsilon

        self._intermediate_size = intermediate_size
        self._intermediate_activation = tf.keras.activations.get(intermediate_activation)
        self._hidden_dropout_rate = hidden_dropout_rate

        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)

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

        self._size_per_head_for_query_and_key = int(hidden_size / self._num_attention_heads)
        common_kwargs = dict(
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint
        )

        # self attention
        self.self_attention = CacheAttention(
            num_attention_heads=self._num_attention_heads,
            size_per_head_for_query_and_key=self._size_per_head_for_query_and_key,
            attention_dropout_rate=self._attention_dropout_rate,
            use_bias=self._use_bias,
            name='self_attention',
            **common_kwargs
        )
        self.self_attention_output_dense = EinsumDense(
            'abc,cd->abd',
            output_shape=(None, hidden_size),
            bias_axes='d',
            name='output',
            **common_kwargs
        )

        self.self_attention_dropout = tf.keras.layers.Dropout(
            rate=self._hidden_dropout_rate
        )
        self.self_attention_layer_norm = tf.keras.layers.LayerNormalization(
            name='self_attention_layer_norm',
            axis=-1,
            epsilon=self._norm_epsilon
        )
        self.encoder_decoder_attention = MultiHeadAttention(
            num_attention_heads=self._num_attention_heads,
            size_per_head_for_query_and_key=self._size_per_head_for_query_and_key,
            size_per_head_for_value=self._size_per_head_for_query_and_key,
            attention_dropout_rate=self._attention_dropout_rate,
            output_shape=hidden_size,
            use_bias=self._use_bias,
            name='attention/encoder_decoder',
            **common_kwargs
        )

        self.encoder_decoder_attention_dropout = tf.keras.layers.Dropout(
            rate=self._hidden_dropout_rate
        )
        self.encoder_decoder_attention_layer_norm = tf.keras.layers.LayerNormalization(
            name='attention/encoder_decoder_output_layer_norm',
            axis=-1,
            epsilon=self._norm_epsilon
        )

        self.intermediate_dense = EinsumDense(
            'abc,cd->abd',
            output_shape=(None, self._intermediate_size),
            bias_axes='d',
            name='intermediate',
            **common_kwargs
        )
        self.intermediate_activation_layer = tf.keras.layers.Activation(
            self._intermediate_activation
        )
        self.intermediate_dropout_layer = tf.keras.layers.Dropout(
            rate=self._hidden_dropout_rate
        )
        self.output_dense = EinsumDense(
            'abc,cd->abd',
            output_shape=(None, hidden_size),
            bias_axes='d',
            name='output',
            **common_kwargs
        )
        self.output_dropout = tf.keras.layers.Dropout(
            rate=self._hidden_dropout_rate
        )
        self.output_layer_norm = tf.keras.layers.LayerNormalization(
            name='output_layer_norm',
            axis=-1,
            epsilon=self._norm_epsilon
        )
        super(TransformerDecoderLayer, self).build(input_shape)

    def call(self, inputs, training, cache=None, decode_loop_step=None):
        targets_tensor, encoder_output, encoder_decoder_attention_mask, self_attention_mask = inputs[:4]
        source_tensor = targets_tensor
        if self._norm_first:
            targets_tensor = self.self_attention_layer_norm(targets_tensor)

        # 新的 cache 会被返回，用于下一轮解码
        self_attention_output, cache = self.self_attention(
            query=targets_tensor,
            value=targets_tensor,
            attention_mask=self_attention_mask,
            cache=cache,
            decode_loop_step=decode_loop_step
        )

        if training:
            self_attention_output = self.self_attention_dropout(self_attention_output)

        if self._norm_first:
            self_attention_output = source_tensor + self_attention_output
        else:
            self_attention_output = self.self_attention_layer_norm(
                targets_tensor + self_attention_output
            )

        if self._norm_first:
            source_self_attention_output = self_attention_output
            self_attention_output = self.encoder_decoder_attention_layer_norm(
                self_attention_output
            )
        encoder_decoder_attention_inputs = dict(
            query=self_attention_output,
            value=encoder_output,
            key=encoder_output,
            attention_mask=encoder_decoder_attention_mask
        )
        attention_output = self.encoder_decoder_attention(
            **encoder_decoder_attention_inputs
        )

        if training:
            attention_output = self.encoder_decoder_attention_dropout(attention_output)

        if self._norm_first:
            attention_output = source_self_attention_output + attention_output
        else:
            attention_output = self.encoder_decoder_attention_layer_norm(
                self_attention_output + attention_output
            )
        if self._norm_first:
            source_self_attention_output = attention_output
            attention_output = self.output_layer_norm(attention_output)

        intermediate_output = self.intermediate_dense(attention_output)
        intermediate_output = self.intermediate_activation_layer(intermediate_output)

        if training:
            intermediate_output = self.intermediate_dropout_layer(intermediate_output)

        layer_output = self.output_dense(intermediate_output)

        if training:
            layer_output = self.output_dropout(layer_output)

        if self._norm_first:
            layer_output = source_self_attention_output + layer_output
        else:
            layer_output = self.output_layer_norm(layer_output + attention_output)

        return layer_output, cache
