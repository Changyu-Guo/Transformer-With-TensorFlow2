# -*- coding: utf - 8 -*-

import tensorflow as tf
from layers.transformer_layers import decoder_layer


class TransformerDecoderStack(tf.keras.layers.Layer):
    
    def __init__(
            self,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=1024,
            intermediate_activation='relu',
            attention_dropout_rate=0.0,
            hidden_dropout_rate=0.0,
            use_bias=False,
            norm_first=True,
            norm_epsilon=1e-6,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs
    ):
        super(TransformerDecoderStack, self).__init__(**kwargs)

        self._num_hidden_layers = num_hidden_layers
        self._num_attention_heads = num_attention_heads
        self._intermediate_size = intermediate_size
        self._intermediate_activation = intermediate_activation
        self._attention_dropout_rate = attention_dropout_rate
        self._hidden_dropout_rate = hidden_dropout_rate
        self._use_bias = use_bias
        self._norm_first = norm_first
        self._norm_epsilon = norm_epsilon
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)

    def build(self, input_shape):

        common_kwargs = dict(
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint
        )

        self.decoder_layers = []
        for i in range(self._num_hidden_layers):
            self.decoder_layers.append(
                decoder_layer.TransformerDecoderLayer(
                    num_attention_heads=self._num_attention_heads,
                    intermediate_size=self._intermediate_size,
                    intermediate_activation=self._intermediate_activation,
                    hidden_dropout_rate=self._hidden_dropout_rate,
                    attention_dropout_rate=self._attention_dropout_rate,
                    use_bias=self._use_bias,
                    norm_first=self._norm_first,
                    norm_epsilon=self._norm_epsilon,
                    **common_kwargs,
                    name=('layer_%d' % i)
                )
            )

        super(TransformerDecoderStack, self).build(input_shape)

    def get_config(self):
        config = {
            'num_hidden_layers': self._num_hidden_layers,
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

        base_config = super(TransformerEncoderStack, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def num_attention_heads(self):
        return self._num_attention_heads

    @property
    def num_hidden_layers(self):
        return self._num_hidden_layers

    def call(self, targets_embeddings, encoder_outputs, padding_mask, look_ahead_mask, training, cache=None):

        decoder_outputs = targets_embeddings

        for i in range(self._num_hidden_layers):
            decoder_inputs = [decoder_outputs, encoder_outputs, padding_mask, look_ahead_mask]
            if cache is None:
                decoder_outputs, _ = self.decoder_layers[i](decoder_inputs)
            else:
                # 对 cache 进行覆盖修改
                cache_layer_idx = str(i)
                decoder_outputs, cache[cache_layer_idx] = self.decoder_layers[i](
                    decoder_inputs,
                    cache=cache[cache_layer_idx]
                )

        return decoder_outputs
