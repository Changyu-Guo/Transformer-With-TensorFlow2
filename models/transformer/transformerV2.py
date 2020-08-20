# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from layers.embedding_layers.word_embedding_share_weights_layer import WordEmbeddingShareWeights
from layers.embedding_layers.relative_position_embedding_layer import RelativePositionEmbedding
from layers.transformer_layers import encoder_layer, decoder_layer
from layers.feed_forward_layers import feed_forward_net_layer
from metrics import transformer_metrics as metrics
from ops import beam_search
from layers import utils


def create_model(params, is_train):
    with tf.name_scope('model'):
        if is_train:
            inputs = tf.keras.layers.Input((None,), dtype=tf.int64, name='inputs_ids')
            targets = tf.keras.layers.Input((None,), dtype=tf.int64, name='targets_ids')
            static_model = Transformer(
                params,
                name='transformer_v2'
            )
            logits = static_model([inputs, targets], training=is_train)
            targets_vocab_size = params['targets_vocab_size']
            label_smoothing = params['label_smoothing']
            if params['enable_metrics_in_training']:
                logits = metrics.MetricLayer(targets_vocab_size)([logits, targets])
            logits = tf.keras.layers.Lambda(
                lambda x: x, name='logits', dtype=tf.float32
            )(logits)
            model = tf.keras.Model([inputs, targets], logits)
            loss = metrics.transformer_loss(logits, targets, label_smoothing, targets_vocab_size)
            model.add_loss(loss)
            return model
        else:
            # (batch_size, seq_len)
            inputs = tf.keras.layers.Input((None,), dtype=tf.int32, name='inputs')
            static_model = Transformer(
                params,
                name='transformer_v2'
            )
            ret = static_model([inputs], training=is_train)
            outputs, scores = ret['outputs'], ret['scores']
            return tf.keras.Model(inputs, [outputs, scores])


class Transformer(tf.keras.Model):
    def __init__(self, params, name=None):
        super(Transformer, self).__init__(name=name)

        self._inputs_vocab_size = params['inputs_vocab_size']
        self._targets_vocab_size = params['targets_vocab_size']
        self._hidden_size = params['hidden_size']
        self._num_hidden_layers = params['num_hidden_layers']
        self._num_attention_heads = params['num_attention_heads']
        self._intermediate_size = params['intermediate_size']
        self._intermediate_activation = utils.get_activation(params['intermediate_activation'])
        self._extra_decode_len = params['extra_decode_len']
        self._hidden_dropout_rate = params['hidden_dropout_rate']
        self._attention_dropout_rate = params['attention_dropout_rate']
        self._norm_first = params['norm_first']
        self._use_bias = params['use_bias']
        self._norm_epsilon = params['norm_epsilon']
        self._dtype = params['dtype']
        self._padded_decode = params['padded_decode']
        self._beam_size = params['beam_size']
        self._alpha = params['alpha']
        self._kernel_initializer = tf.keras.initializers.get(params['kernel_initializer'])
        self._bias_initializer = tf.keras.initializers.get(params['bias_initializer'])
        self._kernel_regularizer = tf.keras.regularizers.get(params['kernel_regularizer'])
        self._bias_regularizer = tf.keras.regularizers.get(params['bias_regularizer'])
        self._activity_regularizer = tf.keras.regularizers.get(params['activity_regularizer'])
        self._kernel_constraint = tf.keras.constraints.get(params['kernel_constraint'])
        self._bias_constraint = tf.keras.constraints.get(params['bias_constraint'])

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
        self.encoder_layers = [self.encoder_layer] * self._num_hidden_layers

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
        self.decoder_layers = [self.decoder_layer] * self._num_hidden_layers

    def call(self, inputs, training):
        if len(inputs) == 2:
            inputs, targets = inputs[0], inputs[1]
        else:
            inputs, targets = inputs[0], None

        with tf.name_scope('transformer'):
            # (batch_size, 1, seq_len)
            inputs_padding_mask = utils.get_padding_mask(inputs)

            # (batch_size, inputs_seq_len, hidden_size)
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
        :param inputs_padding_mask: (batch_size, 1, inputs_seq_len)
        :param training: boolean
        :return: (batch_size, inputs_seq_len, hidden_size)
        """
        with tf.name_scope('encode'):

            # (batch_size, inputs_seq_len, hidden_size)
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

                layer_name = 'layer_%d' % n
                with tf.name_scope(layer_name):
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
        """
        :param encoder_outputs: (batch_size, inputs_seq_len, hidden_size)
        :param encoder_decoder_attention_mask: (batch_size, 1, seq_len)
        :param training:
        :return:
        """

        encoder_outputs = tf.cast(encoder_outputs, self._dtype)
        if self._padded_decode:
            batch_size = encoder_outputs.shape.as_list()[0]
            inputs_seq_len = encoder_outputs.shape.as_list()[1]
        else:
            batch_size = tf.shape(encoder_outputs)[0]
            inputs_seq_len = tf.shape(encoder_outputs)[1]

        max_decode_len = inputs_seq_len + self._extra_decode_len
        encoder_decoder_attention_mask = tf.cast(encoder_decoder_attention_mask, self._dtype)

        auto_regressive_decode_fn = self._get_auto_regressive_decode_fn(
            max_decode_len,
            training=training
        )

        # 初始只有一个 [PAD]
        # (batch_size, 1)
        initial_ids = tf.zeros(shape=(batch_size, 1), dtype=tf.int32)
        # 0: 未解码任何一个字符
        init_decode_len = max_decode_len if self._padded_decode else 0
        size_per_head = self._hidden_size // self._num_attention_heads

        cache = {
            'layer_%d' % layer: {
                'key': tf.zeros(
                    [batch_size, init_decode_len, self._num_attention_heads, size_per_head],
                    dtype=self._dtype
                ),
                'value': tf.zeros(
                    [batch_size, init_decode_len, self._num_attention_heads, size_per_head],
                    dtype=self._dtype
                )
            } for layer in range(self._num_hidden_layers)
        }

        cache['encoder_outputs'] = encoder_outputs
        cache['encoder_decoder_attention_mask'] = encoder_decoder_attention_mask

        decoded_ids, scores = beam_search.sequence_beam_search(
            symbols_to_logits_fn=auto_regressive_decode_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self._targets_vocab_size,
            beam_size=self._beam_size,
            alpha=self._alpha,
            max_decode_length=max_decode_len,
            eos_id=EOS_ID,
            padded_decode=self._padded_decode,
            dtype=self._dtype
        )

        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        return {
            'outputs': top_decoded_ids,
            'scores': top_scores
        }

    def _get_auto_regressive_decode_fn(self, max_decode_len, training):
        # (max_decode_len + 1, hidden_size)
        timing_signal = self.position_embedding(
            inputs=None, length=max_decode_len + 1
        )
        timing_signal = tf.cast(timing_signal, self._dtype)

        # (max_decode_len, max_decode_len)
        targets_look_ahead_mask = utils.get_look_ahead_mask(
            max_decode_len,
            self._dtype
        )

        def auto_regressive_decode_fn(ids, i, cache):
            """
            :param ids: (batch_size * beam_size, i + 1)
            :param i: 解码第一个字符的时候 i = 0
            :param cache:
            :return:
            """

            # 取出前一个字符
            # (batch_size * beam_size, 1)
            decoder_input = ids[:, -1:]

            # (batch_size * beam_size, 1, hidden_size)
            decoder_input = self.targets_embedding_softmax_layer(decoder_input)

            # 位置编码
            decoder_input += timing_signal[i: i + 1]

            # 取出当前位置的 mask
            self_attention_mask = targets_look_ahead_mask[i:i + 1, :i + 1]

            decoder_outputs = decoder_input
            for n, layer in enumerate(self.decoder_layers):

                layer_name = 'layer_%d' % n
                layer_cache = cache[layer_name]
                with tf.name_scope(layer_name):
                    decoder_outputs, _ = layer(
                        inputs=[
                            decoder_outputs,
                            cache.get('encoder_outputs'),
                            cache.get('encoder_decoder_attention_mask'),
                            self_attention_mask
                        ],
                        training=training,
                        cache=layer_cache,
                        decode_loop_step=None
                    )
            logits = self.targets_embedding_softmax_layer(decoder_outputs, mode='linear')
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        return auto_regressive_decode_fn

    def get_config(self):
        return {
            'params': self.params
        }
