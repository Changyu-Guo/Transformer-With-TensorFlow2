# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from models.transformer.transformerV2 import create_model
from models.transformer.transformer_params import PARAMS


class TransformerV2Test(tf.test.TestCase):

    def setUp(self):
        self.params = params = PARAMS
        params['train_batch_size'] = 16
        params['use_synthetic_data'] = True
        params['hidden_size'] = 12
        params['num_hidden_layers'] = 2
        params['intermediate_size'] = 14
        params['num_attention_heads'] = 2
        params['inputs_vocab_size'] = 41
        params['targets_vocab_size'] = 61
        params['extra_decode_len'] = 2
        params['beam_size'] = 3
        params['dtype'] = tf.float32

    def test_create_model_train(self):
        model = create_model(self.params, is_train=True)
        inputs, outputs = model.inputs, model.outputs
        self.assertEqual(len(inputs), 2)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(inputs[0].shape.as_list(), [None, None])
        self.assertEqual(inputs[0].dtype, tf.int64)
        self.assertEqual(inputs[1].shape.as_list(), [None, None])
        self.assertEqual(inputs[1].dtype, tf.int64)
        self.assertEqual(outputs[0].shape.as_list(), [None, None, 61])
        self.assertEqual(outputs[0].dtype, tf.float32)

    def test_create_model_not_train(self):
        model = create_model(self.params, False)
        inputs, outputs = model.inputs, model.outputs
        self.assertEqual(len(inputs), 1)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(inputs[0].shape.as_list(), [None, None])
        self.assertEqual(inputs[0].dtype, tf.int64)
        self.assertEqual(outputs[0].shape.as_list(), [None, None])
        self.assertEqual(outputs[0].shape.dtype, tf.int32)
        self.assertEqual(outputs[1].shape.as_list(), [None])
        self.assertEqual(outputs[1].dtype, tf.float32)


if __name__ == '__main__':
    tf.test.main()
