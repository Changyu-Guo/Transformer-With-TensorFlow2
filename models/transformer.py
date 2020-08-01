# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from layers import attention_layers, position_embedding, embedding_layers, feed_forward_net_layers, utils


def create_model(params, is_train):
    with tf.name_scope('model'):
        if is_train:

            # [batch_size, inputs_seq_len]
            inputs = tf.keras.layers.Input((None,), dtype='int64', name='inputs')

            # [batch_size, targets_seq_len]
            targets = tf.keras.layers.Input((None,), dtype='int64', name='targets')

            # model
            internal_model = Transformer(params, name='transformer')

            # forward pass
            logits = internal_model([inputs, targets], training=is_train)

            # keras model
            model = tf.keras.Model([inputs, targets], logits)
            return model

        else:

            # [batch_size, inputs_seq_len]
            inputs = tf.keras.layers.Input((None,), dtype=tf.int64, name='inputs')

            # model
            internal_model = Transformer(params, name='transformer')

            # forward pass
            ret = internal_model([inputs], training=is_train)
            outputs, scores = ret['outputs'], ret['scores']

            # keras model
            return tf.keras.Model(inputs, [outputs, scores])


class Transformer(tf.keras.Model):
    def __init__(self, params, name=None):
        """
        :param params: vocab_size, hidden_size
        :param name: model name
        """
        super(Transformer, self).__init__(name=name)
        self.params = params

        # 使用参数初始化各层
        self.embedding_softmax_layer = embedding_layers.WordEmbeddingShareWeights(
            params['vocab_size'], params['hidden_size']
        )
        self.position_embedding = position_embedding.RelativePositionEmbedding(
            hidden_size=params['hidden_size']
        )
        self.encoder_stack = EncoderStack(params)
        self.decoder_stack = DecoderStack(params)

    def get_config(self):
        return {
            'params': self.params
        }

    def call(self, inputs, training):
        """
            Compute targets logits or infer targets sequences.
        Args:
            inputs: a list such as [inputs] or [inputs, targets]
            training: training mode or not

        Returns:
            if inputs = [inputs, targets], return logits for each word in
            targets sequence, the shape is [batch_size, target_seq_len, vocab_size]

            if inputs = [inputs], generate targets language sequence one token
            at a time. returns a dict:
            {
                outputs: [batch_size, output_seq_len],
                scores: [batch_size, float]
            }
        """

        # encode and decode
        if len(inputs) == 2:
            inputs, targets = inputs[0], inputs[1]

        # encode and predict
        else:
            inputs, targets = inputs[0], None

        with tf.name_scope('Transformer'):
            # 一定要在这里获取，因为 decode 或 predict 的时候不会把 inputs 传入函数
            # 因此无法在 decode 和 predict 函数里获取 inputs 的 padding_mask
            # shape: [batch_size, 1, 1, seq_len]
            inputs_padding_mask = utils.get_attention_padding_mask(inputs)

            # encode
            # shape: [batch_size, seq_len, hidden_size]
            encoder_outputs = self.encode(inputs, inputs_padding_mask, training)
            # decode or predict
            if targets is None:
                # shape: []
                return self.predict(encoder_outputs, inputs_padding_mask, training)
            else:
                # [batch_size, seq_len, vocab_size]
                logits = self.decode(targets, encoder_outputs, inputs_padding_mask, training)
                return logits

    def encode(self, inputs, inputs_padding_mask, training):
        """
        :param inputs: a list of each item shape as [batch_size, seq_len]
        :param inputs_padding_mask: [batch_size, 1, 1, seq_len]
        :param training: boolean
        :return:
        """
        with tf.name_scope('encode'):

            # [batch_size, seq_len, hidden_size]
            embedded_inputs = self.embedding_softmax_layer(inputs, mode='embedding')
            embedded_inputs = tf.cast(embedded_inputs, self.params['dtype'])

            # convert data type
            # [batch_size, 1, 1, seq_len]
            inputs_padding_mask = tf.cast(inputs_padding_mask, self.params['dtype'])

            with tf.name_scope('add_pos_encoding'):
                # [seq_len, hidden_size]
                pos_encoding = self.position_embedding(inputs=embedded_inputs)
                pos_encoding = tf.cast(pos_encoding, self.params['dtype'])

                # pos_encoding 会广播到每个 batch
                # [batch_size, seq_len, hidden_size]
                encoder_inputs = embedded_inputs + pos_encoding

            if training:
                # 词嵌入 + 位置编码之后 Dropout 一次
                encoder_inputs = tf.nn.dropout(
                    encoder_inputs, rate=self.params['layer_postprocess_dropout']
                )
            return self.encoder_stack(
                encoder_inputs, inputs_padding_mask, training=training
            )

    def decode(self, targets, encoder_outputs, inputs_padding_mask, training):
        with tf.name_scope('decode'):
            # embedding
            decoder_inputs = self.embedding_softmax_layer(targets)
            decoder_inputs = tf.cast(decoder_inputs, self.params['dtype'])

            inputs_padding_mask = tf.cast(inputs_padding_mask, self.params['dtype'])

            # shift
            with tf.name_scope('shift_targets'):
                # 在每个 seq 前面插入一个 [PAD], 表示句子开始
                # 然后将除了最后一个字符的句子取出来
                decoder_inputs = tf.pad(decoder_inputs, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]

            with tf.name_scope('add_pos_encoding'):
                pos_encoding = self.position_embedding(decoder_inputs)
                pos_encoding = tf.cast(pos_encoding, self.params['dtype'])
                decoder_inputs += pos_encoding

            if training:
                decoder_inputs = tf.nn.dropout(
                    decoder_inputs, rate=self.params['layer_postprocess_dropout']
                )

            seq_len = tf.shape(decoder_inputs)[1]
            targets_look_ahead_mask = utils.get_look_ahead_mask(
                seq_len, dtype=self.params['dtype']
            )

            outputs = self.decoder_stack(
                decoder_inputs,
                encoder_outputs,
                targets_look_ahead_mask,
                inputs_padding_mask,
                training=training
            )

            logits = self.embedding_softmax_layer(outputs, mode='linear')
            logits = tf.cast(logits, tf.float32)
            return logits


class PrePostProcessingWrapper(tf.keras.layers.Layer):
    def __init__(self, layer, params):
        super(PrePostProcessingWrapper, self).__init__()
        self.layer = layer
        self.params = params
        self.postprocess_dropout = params['layer_postprocess_dropout']

    def build(self, input_shape):
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, dtype='float32'
        )
        super(PrePostProcessingWrapper, self).build(input_shape)

    def get_config(self):
        return {
            'params': self.params
        }

    def call(self, x, *args, **kwargs):
        """
        :param x: for attention, x is query, and final query will add to attention_output
        """
        training = kwargs['training']

        # layer normalization
        y = self.layer_norm(x)

        # attention or feed forward net
        y = self.layer(y, *args, **kwargs)

        # dropout
        if training:
            y = tf.nn.dropout(y, rate=self.postprocess_dropout)

        # res add
        return x + y


class EncoderStack(tf.keras.layers.Layer):
    def __init__(self, params):
        super(EncoderStack, self).__init__()
        self.params = params
        self.layers = []

    def build(self, input_shape):
        params = self.params
        for _ in range(params['num_hidden_layers']):
            self_attention = attention_layers.MultiHeadAttention(
                num_heads=params['num_heads'],
                key_size=params['hidden_size'],
                drop_rate=params['attention_dropout']
            )
            feed_forward_net = feed_forward_net_layers.FeedForwardNetwork(
                filter_size=params['filter_size'],
                hidden_size=params['hidden_size'],
                drop_rate=params['relu_dropout']
            )
            self.layers.append([
                PrePostProcessingWrapper(self_attention, params),
                PrePostProcessingWrapper(feed_forward_net, params)
            ])

        self.output_normalization = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, dtype=tf.float32
        )
        super(EncoderStack, self).build(input_shape)

    def get_config(self):
        return {
            'params': self.params
        }

    def call(self, encoder_inputs, attention_mask, training):
        """
        :param encoder_inputs: [batch_size, seq_len, hidden_size]
        :param attention_mask: [batch_size, 1, 1, seq_len]
        :param training: boolean
        :return: [batch_size, seq_len, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            self_attention = layer[0]
            feed_forward_net = layer[1]

            with tf.name_scope('layer_%d' % n):
                # compute pipeline
                with tf.name_scope('self_attention'):
                    encoder_outputs = self_attention(
                        encoder_inputs,
                        encoder_inputs,
                        encoder_inputs,
                        attention_mask,
                        training=training
                    )
                with tf.name_scope('feed_forward_net'):
                    encoder_outputs = feed_forward_net(
                        encoder_outputs, training=training
                    )

        return self.output_normalization(encoder_outputs)


class DecoderStack(tf.keras.layers.Layer):
    def __init__(self, params):
        super(DecoderStack, self).__init__()
        self.params = params
        self.layers = []

    def build(self, input_shape):
        params = self.params
        for _ in range(params['num_hidden_layers']):
            self_attention = attention_layers.MultiHeadAttention(
                num_heads=params['num_heads'],
                key_size=params['hidden_size'],
                drop_rate=params['attention_dropout']
            )
            encoder_decoder_attention = attention_layers.MultiHeadAttention(
                num_heads=params['num_heads'],
                key_size=params['hidden_size'],
                drop_rate=params['attention_dropout']
            )
            feed_forward_net = feed_forward_net_layers.FeedForwardNetwork(
                filter_size=params['filter_size'],
                hidden_size=params['hidden_size'],
                drop_rate=params['relu_dropout']
            )
            self.layers.append([
                PrePostProcessingWrapper(self_attention, params),
                PrePostProcessingWrapper(encoder_decoder_attention, params),
                PrePostProcessingWrapper(feed_forward_net, params)
            ])
        
        self.output_normalization = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, dtype=tf.float32
        )
        super(DecoderStack, self).build(input_shape)

    def get_config(self):
        return {
            'params': self.params
        }

    def call(self, decoder_inputs, encoder_outputs, targets_look_ahead_mask, inputs_padding_mask,
            training, cache=None, decode_loop_step=None):
        for n, layer in enumerate(self.layers):
            self_attention = layer[0]
            encoder_decoder_attention = layer[1]
            feed_forward_net = layer[2]

            layer_name = 'layer_%d' % n
            layer_cache = cache[layer_name] if cache is not None else None

            with tf.name_scope(layer_name):
                with tf.name_scope('self_attention'):
                    decoder_outputs = self_attention(
                        decoder_inputs,
                        decoder_inputs,
                        decoder_inputs,
                        targets_look_ahead_mask,
                        training=training
                    )
                with tf.name_scope('encoder_decoder_attention'):
                    decoder_outputs = encoder_decoder_attention(
                        decoder_outputs,
                        encoder_outputs,
                        encoder_outputs,
                        inputs_padding_mask,
                        training=training
                    )
                with tf.name_scope('feed_forward_net'):
                    decoder_outputs = feed_forward_net(decoder_outputs, training=training)

        return self.output_normalization(decoder_outputs)
