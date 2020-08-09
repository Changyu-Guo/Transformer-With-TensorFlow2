# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from layers.embedding_layers.word_embedding_share_weights_layer import WordEmbeddingShareWeights
from layers.embedding_layers.relative_position_embedding_layer import RelativePositionEmbedding
from layers.transformer_layers import encoder_layer, decoder_layer
from layers.feed_forward_layers import feed_forward_net_layer
from layers import utils


class Transformer(tf.keras.Model):
    def __init__(
            self,
            inputs_vocab_size,
            targets_vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            intermediate_activation,
            max_decode_len,
            hidden_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            norm_first=False,
            use_bias=True,
            norm_epsilon=1e-12,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name=None,
            **kwargs
    ):
        super(Transformer, self).__init__(name=name, **kwargs)

        self._inputs_vocab_size = inputs_vocab_size
        self._targets_vocab_size = targets_vocab_size
        self._hidden_size = hidden_size
        self._num_hidden_layers = num_hidden_layers
        self._num_attention_heads = num_attention_heads
        self._intermediate_size = intermediate_size
        self._intermediate_activation = tf.keras.activations.get(intermediate_activation)
        self._max_decode_len = max_decode_len
        self._hidden_dropout_rate = hidden_dropout_rate
        self._attention_dropout_rate = attention_dropout_rate
        self._norm_first = norm_first
        self._use_bias = use_bias
        self._norm_epsilon = norm_epsilon
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)
    
    def build(self, input_shape):

        # embeddings
        self.inputs_embedding_softmax_layer = WordEmbeddingShareWeights(
            vocab_size=self._inputs_vocab_size, hidden_size=self._hidden_size
        )
        self.targets_embedding_softmax_layer = WordEmbeddingShareWeights(
            vocab_size=self._targets_vocab_size, hidden_size=self._hidden_size
        )
        self.position_embedding = RelativePositionEmbedding(
            hidden_size=self._hidden_size
        )

        # common args
        common_args = dict(
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint
        )

        # encoder
        self.encoder_layer = encoder_layer.TransformerEncoderLayer(
            num_attention_heads=self._num_attention_heads,
            intermediate_size=self._intermediate_size,
            intermediate_activation=self._intermediate_activation,
            hidden_dropout_rate=self._hidden_dropout_rate,
            attention_dropout_rate=self._attention_dropout_rate,
            use_bias=self._use_bias,
            norm_first=self._norm_first,
            norm_epsilon=self._norm_first,
            **common_args
        )
        self.encoder_layers = [self.encoder_layer] * num_hidden_layers

        # decoder
        self.decoder_layer = decoder_layer.TransformerDecoderLayer(
            num_attention_heads=self._num_attention_heads,
            intermediate_size=self._intermediate_size,
            intermediate_activation=self._intermediate_activation,
            hidden_dropout_rate=self._hidden_dropout_rate,
            attention_dropout_rate=self._attention_dropout_rate,
            use_bias=self._use_bias,
            norm_first=self._norm_first,
            norm_epsilon=self._norm_epsilon,
            **common_args
        )
        self.decoder_layers = [self.decoder_layer] * num_hidden_layers

        super(Transformer, self).build(input_shape)

    def call(self, inputs, training):
        if len(inputs) == 2:
            inputs, targets = inputs[0], inputs[1]
        else:
            inputs, targets = inputs[0], None

        with tf.name_scope('transformer'):
            inputs_padding_mask = utils.get_attention_padding_mask(inputs)

            encoder_outputs = self.encode(inputs, inputs_padding_mask, training)

            if targets is None:
                return self.predict(encoder_outputs, inputs_padding_mask, training)
            else:
                decoder_outputs = self.decode(targets, encoder_outputs, inputs_padding_mask, training)
                logits = self.targets_embedding_softmax_layer(decoder_outputs, mode='linear')
                return logits

    def encode(self, inputs, inputs_padding_mask, training):
        """
        :param inputs: (batch_size, inputs_seq_len)
        :param inputs_padding_mask: (batch_size, inputs_seq_len)
        :param training: boolean
        :return: (batch_size, inputs_seq_len, hidden_size)
        """
        with tf.name_scope('encode'):

            embedded_inputs = self.inputs_embedding_softmax_layer(
                inputs, mode='embedding'
            )

            with tf.name_scope('position_encoding'):
                position_encoding = self.position_embedding(inputs=embedded_inputs)

                encoder_inputs = embedded_inputs + position_encoding

            if training:
                encoder_inputs = tf.nn.dropout(
                    encoder_inputs, rate=self._hidden_dropout_rate
                )

            encoder_outputs = encoder_inputs
            for n, layer in enumerate(self.encoder_layers):

                with tf.name_scope('layer_%d' % n):
                    encoder_outputs = layer(
                        inputs=[encoder_outputs, inputs_padding_mask],
                        training=training
                    )

            return encoder_outputs

    def decode(
            self,
            targets,
            encoder_outputs,
            inputs_padding_mask,
            training
    ):
        """
        :param targets: (batch_size, targets_seq_len)
        :param encoder_outputs: (batch_size, inputs_seq_len, hidden_size)
        :param inputs_padding_mask: (batch_size, inputs_seq_len)
        :param training: boolean
        :return: (batch_size, targets_seq_len, vocab_size)
        """

        with tf.name_scope('decode'):

            decoder_inputs = self.targets_embedding_softmax_layer(
                targets,
                mode='embedding'
            )

            with tf.name_scope('shift_targets'):
                decoder_inputs = tf.pad(
                    decoder_inputs,
                    [[0, 0], [1, 0], [0, 0]]
                )[:, :-1, :]

            with tf.name_scope('position_encoding'):
                position_encoding = self.position_embedding(decoder_inputs)
                decoder_inputs += position_encoding

            if training:
                decoder_inputs = tf.nn.dropout(
                    decoder_inputs, rate=self._hidden_dropout_rate
                )

            targets_seq_len = tf.shape(decoder_inputs)[1]
            targets_look_ahead_mask = utils.get_look_ahead_mask(targets_seq_len)

            decoder_outputs = decoder_inputs
            for n, layer in enumerate(self.decoder_layers):

                with tf.name_scope('layer_%d' % n):
                    decoder_outputs, _ = layer(
                        inputs=[
                            decoder_outputs,
                            encoder_outputs,
                            inputs_padding_mask,
                            targets_look_ahead_mask
                        ],
                        training=training
                    )

            return decoder_outputs

    def predict(self, encoder_outputs, encoder_decoder_attention_mask, training):

        batch_size = tf.shape(encoder_outputs)[0]
        inputs_seq_len = tf.shape(encoder_outputs)[1]

    def _get_symbols_to_logits_fn(self, training):
        timing_signal = self.position_embedding(
            inputs=None, length=self._max_decode_len + 1
        )
        targets_padding_mask = utils.get_attention_padding_mask(
            self._max_decode_len
        )

        def auto_regressive_decode_fn(ids, i, cache):
            """
            :param ids: (batch_size * beam_size, i + 1)
            :param i:
            :param cache:
            :return:
            """
            decoder_input = ids[:, -1:]
            decoder_input = self.targets_embedding_softmax_layer(decoder_input)
