# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import position_embedding
import attention
import embedding_layer
import feed_forward_net_layer
import utils


def create_model(params, is_train):
    with tf.name_scope('model'):
        if is_train:
            # [batch_size, seq_len] -> [None, None]
            inputs = tf.keras.layers.Input((None,), dtype='int64', name='inputs')
            target = tf.keras.layers.Input((None,), dtype='int64', name='targets')
            internal_model = Transformer(params, name='transformer')
            logits = internal_model([inputs, targets], training=is_train)


class Transformer(tf.keras.Model):
    def __init__(self, params, name=None):
        """
        :param params: vocab_size, hidden_size
        :param name: model name
        """
        super(Transformer, self).__init__(name=name)
        self.params = params
        self.embedding_softmax_layer = embedding_layer.EmbeddingShareWeights(
            params['vocab_size'], params['hidden_size']
        )
        self.encoder_stack = EncoderStack(params)
        self.decoder_stack = DecoderStack(params)
        self.position_embedding = position_embedding.RelativePositionEmbedding(
            hidden_size=self.params['hidden_size']
        )

    def get_config(self):
        return {
            'params': self.params
        }

    def call(self, inputs, training):
        """
            Compute target logits or infer target sequences.
        Args:
            inputs: a list such as [inputs] or [inputs, target]
            training: training mode or not

        Returns:
            if inputs = [inputs, target], return logits for each word in
            target sequence, the shape is [batch_size, target_seq_len, vocab_size]

            if inputs = [inputs], generate target language sequence one token
            at a time. returns a dict:
            {
                outputs: [batch_size, output_seq_len],
                scores: [batch_size, float]
            }
        """
        if len(inputs) == 2:
            inputs, target = inputs[0], inputs[1]
        else:
            inputs, targets = inputs[0], None
            if self.params['padding_decode']:
                if not self.params['num_replicas']:
                    raise NotImplementedError(
                        "Padded decoding on CPU/GPUs is not supported."
                    )
            decode_batch_size = int(self.params['decode_batch_size'])
            inputs.set_shape([
                decode_batch_size, self.params['decode_max_length']
            ])

        with tf.name_scope('Transformer'):
            attention_bias = utils.get_padding_bias(inputs)
            encoder_outputs = self.encode(inputs, attention_bias, training)
            if target is None:
                return self.predict(encoder_outputs, attention_bias, training)
            else:
                logits = self.decode(targets, encoder_outputs, attention_bias, training)
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
        super(PrePostProcessingWrapper, self).build()

    def get_config(self):
        return {
            'params': self.params
        }

    def call(self, x, *args, **kwargs):
        training = kwargs['training']
        y = self.layer_norm(x)
        y = self.layer(y, *args, **kwargs)
        if training:
            y = tf.nn.dropout(y, rate=self.postprocess_dropout)
        return x + y


class EncoderStack(tf.keras.layers.Layer):
    def __init__(self, params):
        super(EncoderStack, self).__init__()
        self.params = params
        self.layers = []

    def build(self, input_shape):
        params = self.params
        for _ in range(params['num_hidden_layers']):
            self_attention = attention.MultiHeadAttention(
                params['hidden_size'], params['num_heads'],
                params['attention_dropout']
            )
            feed_forward_net = feed_forward_net_layer.FeedForwardNetwork(
                params['hidden_size'], params['filter_size'], params['relu_dropout']
            )
            self.layers.append([
                PrePostProcessingWrapper(self_attention, params),
                PrePostProcessingWrapper(feed_forward_net, params)
            ])

            self.output_normalization = tf.keras.layers.LayerNormalization(
                epsilon=1e-6, dtype='float32'
            )
            super(EncoderStack, self).build(input_shape)

    def get_config(self):
        return {
            'params': self.params
        }

    def call(self, encoder_inputs, attention_bias, inputs_padding, training):
        for n, layer in enumerate(self.layers):
            self_attention = layer[0]
            feed_forward_net = layer[1]

            with tf.name_scope('layer_%d' % n):
                with tf.name_scope('self_attention'):
                    encoder_inputs = self_attention(
                        encoder_inputs, attention_bias, training=training
                    )
                with tf.name_scope('feed_forward_net'):
                    encoder_inputs = feed_forward_net(
                        encoder_inputs, training=training
                    )

        return self.output_normalization(encoder_inputs)


class DecoderStack(tf.keras.layers.Layer):
    def __init__(self, params):
        super(DecoderStack, self).__init__()
        self.params = params
        self.layers = []