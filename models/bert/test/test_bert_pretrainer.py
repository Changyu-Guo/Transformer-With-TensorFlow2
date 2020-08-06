# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models.bert.bert_pretrainer import BertPretrainer
from networks.encoders.bert_encoder import BertEncoder


class BertPretrainerTest(tf.test.TestCase):
    def test_bert_pretrainer(self):
        vocab_size = 100
        seq_len = 512
        test_network = BertEncoder(
            vocab_size=vocab_size,
            num_layers=2,
            max_seq_len=seq_len
        )

        num_classes = 3
        num_token_predictions = 2
        bert_trainer_model = BertPretrainer(
            test_network,
            num_classes=num_classes,
            num_token_predictions=num_token_predictions
        )

        word_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32)
        mask = tf.keras.Input(shape=(seq_len,), dtype=tf.int32)
        type_ids = tf.keras.Input(shape=(seq_len,), dtype=tf.int32)

        masked_lm_positions = tf.keras.Input(
            shape=(num_token_predictions,), dtype=tf.int32
        )

        outputs = bert_trainer_model(
            [word_ids, mask, type_ids, masked_lm_positions]
        )

        expected_lm_shape = [None, num_token_predictions, vocab_size]
        expected_classification_shape = [None, num_classes]
        self.assertAllEqual(expected_lm_shape, outputs['masked_lm'].shape.as_list())
        self.assertAllEqual(expected_classification_shape, outputs['classification'].shape.as_list())

    def test_bert_trainer_tensor_call(self):
        test_network = BertEncoder(
            vocab_size=100, num_layers=2, max_seq_len=2
        )

        bert_trainer_model = BertPretrainer(
            test_network, num_classes=2, num_token_predictions=2
        )

        word_ids = tf.constant([[1, 1], [2, 2]], dtype=tf.int32)
        mask = tf.constant([[1, 1], [1, 0]], dtype=tf.int32)
        type_ids = tf.constant([[1, 1], [2, 2]], dtype=tf.int32)
        lm_mask = tf.constant([[1, 1], [1, 0]], dtype=tf.int32)

        _ = bert_trainer_model([word_ids, mask, type_ids, lm_mask])

    def test_serialize_deserialize(self):
        test_network = BertEncoder(
            vocab_size=100, num_layers=2, max_seq_len=5)

        bert_trainer_model = BertPretrainer(
            test_network, num_classes=4, num_token_predictions=3)

        config = bert_trainer_model.get_config()
        new_bert_trainer_model = BertPretrainer.from_config(config)

        _ = new_bert_trainer_model.to_json()

        self.assertAllEqual(
            bert_trainer_model.get_config(),
            new_bert_trainer_model.get_config()
        )


if __name__ == '__main__':
    tf.test.main()
