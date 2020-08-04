# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.bert import create_bert_models
from models.bert.configs import BertConfig


class BertModelsTest(tf.test.TestCase):
    def setUp(self):
        super(BertModelsTest, self).setUp()
        self._bert_test_config = BertConfig(
            attention_probs_dropout_prob=0.0,
            hidden_act='relu',
            hidden_dropout_prob=0.0,
            hidden_size=16,
            initializer_range=0.02,
            intermediate_size=32,
            max_position_embeddings=128,
            num_attention_heads=2,
            num_hidden_layers=2,
            type_vocab_size=2,
            vocab_size=30522
        )

    def test_classifier_model(self):
        model, core_model = create_bert_models.create_classifier_model(
            self._bert_test_config,
            num_labels=3,
            max_seq_len=5,
            final_layer_initializer=None
        )
        self.assertIsInstance(model, tf.keras.Model)
        self.assertIsInstance(core_model, tf.keras.Model)

        self.assertEqual(model.output.shape.as_list(), [None, 3])

        self.assertIsInstance(core_model.output, list)
        self.assertLen(core_model.output, 2)
        self.assertEqual(core_model.output[0].shape.as_list(), [None, None, 16])
        self.assertEqual(core_model.output[1].shape.as_list(), [None, 16])


if __name__ == '__main__':
    tf.test.main()