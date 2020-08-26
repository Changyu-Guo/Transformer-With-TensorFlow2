# -*- coding: utf - 8 -*-

import numpy as np
import tensorflow as tf
from absl import logging
from absl.testing import parameterized
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from models.transformer import transformer
from layers.transformer_layers.encoder_stack import TransformerEncoderStack
from layers.transformer_layers.decoder_stack import TransformerDecoderStack
from models.transformer import model_params


class TransformerTest(tf.test.TestCase, parameterized.TestCase):

    def test_create_model(self):
        self.params = model_params.TINY_PARAMS
        self.params['batch_size'] = 2
        self.params['hidden_size'] = 12
        self.params['num_hidden_layers'] = 2
        self.params['intermediate_size'] = 14
        self.params['num_attention_heads'] = 2
        self.params['inputs_vocab_size'] = 41
        self.params['targets_vocab_size'] = 42
        self.params['extra_decode_len'] = 2
        self.params['beam_size'] = 3
        self.params['dtype'] = tf.float32

        model = transformer.create_model(self.params, is_train=True)
        inputs, outputs = model.inputs, model.outputs
        self.assertLen(inputs, 2)
        self.assertLen(outputs, 1)
        self.assertEqual(inputs[0].shape.as_list(), [None, None])
        self.assertEqual(inputs[0].dtype, tf.int64)
        self.assertEqual(inputs[1].shape.as_list(), [None, None])
        self.assertEqual(inputs[1].dtype, tf.int64)
        self.assertEqual(outputs[0].shape.as_list(), [None, None, 42])
        self.assertEqual(outputs[0].dtype, tf.float32)

        model = transformer.create_model(self.params, is_train=False)
        inputs, outputs = model.inputs, model.outputs
        self.assertLen(inputs, 1)
        self.assertLen(outputs, 2)
        self.assertEqual(inputs[0].shape.as_list(), [None, None])
        self.assertEqual(inputs[0].dtype, tf.int64)
        self.assertEqual(outputs[0].shape.as_list(), [None, None])
        self.assertEqual(outputs[0].dtype, tf.int32)
        self.assertEqual(outputs[1].shape.as_list(), [None])
        self.assertEqual(outputs[1].dtype, tf.float32)

    def _build_model(self, max_decode_len):
        num_hidden_layers = 1
        num_attention_heads = 2
        intermediate_size = 32
        inputs_vocab_size = 100
        targets_vocab_size = 101
        hidden_size = 16
        encoder_decoder_kwargs = dict(
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            intermediate_activation='relu',
            hidden_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            use_bias=False,
            norm_first=True,
            norm_epsilon=1e-6
        )
        encoder_stack = TransformerEncoderStack(**encoder_decoder_kwargs)
        decoder_stack = TransformerDecoderStack(**encoder_decoder_kwargs)

        return transformer.Transformer(
            inputs_vocab_size=inputs_vocab_size,
            targets_vocab_size=targets_vocab_size,
            hidden_size=hidden_size,
            hidden_dropout_rate=0.01,
            max_decode_len=max_decode_len,
            beam_size=4,
            alpha=0.6,
            encoder_stack=encoder_stack,
            decoder_stack=decoder_stack
        )


if __name__ == '__main__':
    tf.test.main()