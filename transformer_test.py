# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model_params
import transformer


class TransformerTest(tf.test.TestCase):
    def setUp(self):
        params = model_params.TINY_PARAMS
        params['batch_size'] = 2
        params['use_synthetic_data'] = True
        params['hidden_size'] = 12
        params['num_hidden_layers'] = 2
        params['filter_size'] = 14
        params['num_heads'] = 2
        params['vocab_size'] = 41
        params['extra_decode_length'] = 2
        params['beam_size'] = 3
        params['dtype'] = tf.float32
        self.params = params

    def test_create_model_and_train(self):
        model = transformer.create_model(self.params, is_train=True)
        inputs, outputs = model.inputs, model.outputs


if __name__ == '__main__':
    tf.test.main()
