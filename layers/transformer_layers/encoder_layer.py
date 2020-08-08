# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from layers.attention_layers.multi_head_attention_layer import MultiHeadAttention
from layers.attention_layers.einsum_dense import EinsumDense


class TransformerEncoderLayer(tf.keras.layers.Layer):
    """
        Transformer 的编码器

        主要包括如下结构：(不考虑 embedding layer，因为不在循环体内部)
            1. self attention layer (self-attention)
                参数：
                    1. num attention heads
                    2. attention dropout rate
                    3. use bias
            2. position-wise feed-forward network
                参数：
                    1. intermediate size / filter size
                    2. intermediate activation
                    3. hidden dropout rate

    """
    def __init__(
            self,
            num_attention_heads,
            intermediate_size,
            intermediate_activation,
            hidden_dropout_rate=0.0,
            attention_dropout_rate=0.0,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
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
        super(TransformerEncoderLayer, self).__init__(**kwargs)

        # attention
        self._num_attention_heads = num_attention_heads
        self._attention_dropout_rate = attention_dropout_rate
        self._use_bias = use_bias
        self._norm_first = norm_first
        self._norm_epsilon = norm_epsilon

        # position-wise feed-forward network
        self._intermediate_size = intermediate_size
        self._intermediate_activation = tf.keras.activations.get(intermediate_activation)
        self._hidden_dropout_rate = hidden_dropout_rate

        # common args
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)

    def build(self, input_shape):
        """
        :param input_shape: 包含 inputs shape 和 mask shape 或仅包含 inputs shape
        其中 inputs shape 应该为 (batch_size, seq_len, hidden_size)
        mask shape 应该为 (batch_size, seq_len, seq_len)
        因为是 self-attention，所以两个 seq_len 相同
        """

        # 获取 inputs shape
        inputs_tensor_shape = input_shape[0] if len(input_shape) == 2 else input_shape
        inputs_tensor_shape = tf.TensorShape(inputs_tensor_shape)
        if len(inputs_tensor_shape) != 3:
            raise ValueError(
                'TransformerEncoderLayer expects a three-dim input of '
                'shape (batch_size, seq_len, hidden_size)'
            )
        batch_size, seq_len, hidden_size = inputs_tensor_shape

        # 获取 attention mask 相关参数
        if len(input_shape) == 2:
            mask_tensor_shape = tf.TensorShape(input_shape[1])
            expected_mask_tensor_shape = tf.TensorShape(
                [batch_size, seq_len, seq_len]
            )
            if not expected_mask_tensor_shape.is_compatible_with(mask_tensor_shape):
                raise ValueError(
                    'When passing a mask tensor to TransformerEncoderLayer, the '
                    'mask tensor must be of shape (batch_size, '
                    'seq_len, seq_len) (here %s). Got a '
                    'mask tensor of shape %s.' %
                    (expected_mask_tensor_shape, mask_tensor_shape)
                )

        # 在 attention 中
        # query 和 key 具有相同的 size per head
        # 而 key 和 value 具有相同的 seq_len
        # 而在 self-attention 中
        # 三者具有相同的 size per head 和 seq_len
        if hidden_size % self._num_attention_heads != 0:
            raise ValueError(
                'The input size (%d) is not a multiple of the number of attention '
                'heads (%d)' % (hidden_size, self._num_attention_heads)
            )
        self._size_per_head = int(hidden_size // self._num_attention_heads)

        # 构建通用参数
        common_kwargs = dict(
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint
        )

        # attention layer
        self.attention_layer = MultiHeadAttention(
            num_attention_heads=self._num_attention_heads,
            size_per_head_for_query_and_key=self._size_per_head,
            size_per_head_for_value=self._size_per_head,
            attention_dropout_rate=self._attention_dropout_rate,
            use_bias=self._use_bias,
            name='self_attention',
            **common_kwargs
        )
        # attention 后接的 dropout
        self.attention_dropout = tf.keras.layers.Dropout(
            rate=self._hidden_dropout_rate
        )
        # attention 后接的 layer norm
        self.attention_layer_norm = tf.keras.layers.LayerNormalization(
            name='self_attention_layer_norm',
            axis=-1,
            epsilon=self._norm_epsilon,
            dtype=tf.float32
        )

        # 全连接神经网络
        self.intermediate_dense = EinsumDense(
            'abc,cd->abd',
            output_shape=(None, self._intermediate_size),
            bias_axes='d',
            name='feed_forward_net',
            **common_kwargs
        )

        policy = tf.keras.mixed_precision.experimental.global_policy()
        if policy.name == 'mixed_bfloat16':
            policy = tf.float32

        # 全连接神经网络的激活函数
        self.intermediate_dense_activation = tf.keras.layers.Activation(
            self._intermediate_activation, dtype=policy
        )
        self.intermediate_dropout_layer = tf.keras.layers.Dropout(
            rate=self._hidden_dropout_rate
        )

        # 输出
        # (全连接神经网络被拆分为了 filter_dense 和 output_dense)
        self.output_dense = EinsumDense(
            'abc,cd->abd',
            output_shape=(None, hidden_size),
            bias_axes='d',
            name='output_dense',
            **common_kwargs
        )
        self.output_dropout = tf.keras.layers.Dropout(
            rate=self._hidden_dropout_rate
        )
        self.output_layer_norm = tf.keras.layers.LayerNormalization(
            name='output_layer_norm',
            axis=-1,
            epsilon=self._norm_epsilon,
            dtype=tf.float32
        )
        super(TransformerEncoderLayer, self).build(input_shape)

    def call(self, inputs):
        # input: (batch_size, seq_len, hidden_size)
        # mask: (batch_size, seq_len, seq_len)
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            inputs_tensor, attention_mask = inputs
        else:
            inputs_tensor, attention_mask = (inputs, None)

        if self._norm_first:
            source_tensor = inputs_tensor  # 保留操作前的数据，用于后面残差连接
            inputs_tensor = self.attention_layer_norm(inputs_tensor)

        # self-attention
        # query = key = value
        attention_output = self.attention_layer(
            query=inputs_tensor,
            value=inputs_tensor,
            key=inputs_tensor,
            attention_mask=attention_mask
        )
        attention_output = self.attention_dropout(attention_output)
        # 如果之前做过 layer norm，这里只需要进行残差连接
        if self._norm_first:
            attention_output = source_tensor + attention_output
        # 否则先残差连接，然后 layer norm
        else:
            attention_output = self.attention_layer_norm(
                target_tensor + attention_output
            )

        if self._norm_first:
            source_attention_output = attention_output
            attention_output = self.output_layer_norm(attention_output)

        feed_forward_net_output = self.intermediate_dense(attention_output)
        feed_forward_net_output = self.intermediate_dense_activation(feed_forward_net_output)

        layer_output = self.output_dense(feed_forward_net_output)
        layer_output = self.output_dropout(layer_output)

        layer_output = tf.cast(layer_output, tf.float32)
        if self._norm_first:
            layer_output = source_attention_output + layer_output
        else:
            layer_output = self.output_layer_norm(layer_output + attention_output)

        return layer_output

    def get_config(self):
        config = {
            'num_attention_heads': self._num_attention_heads,
            'intermedia_size': self._intermediate_size,
            'intermediate_activation': self._intermediate_activation,
            'hidden_dropout_rate': self._hidden_dropout_rate,
            'attention_dropout_rate': self._attention_dropout_rate,
            'kernel_initializer': tf.keras.initializers.serialize(self._kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self._bias_initializer),
            'kernel_regularizer':
                tf.keras.regularizers.serialize(self._kernel_regularizer),
            'bias_regularizer':
                tf.keras.regularizers.serialize(self._bias_regularizer),
            'activity_regularizer':
                tf.keras.regularizers.serialize(self._activity_regularizer),
            'kernel_constraint':
                tf.keras.constraints.serialize(self._kernel_constraint),
            'bias_constraint':
                tf.keras.constraints.serialize(self._bias_constraint),
            'use_bias': self._use_bias,
            'norm_first': self._norm_first,
            'norm_epsilon': self._norm_epsilon
        }
        base_config = super(TransformerEncoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
