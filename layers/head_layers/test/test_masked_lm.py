# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from layers.head_layers import masked_lm
from networks.encoders import bert_encoder


class MaskedLMTest(tf.test.TestCase):
    def create_layer(
            self,
            vocab_size,
            hidden_size,
            output='predictions',
            xformer_stack=None
    ):
        if xformer_stack is None:
            xformer_stack = bert_encoder.BertEncoder(
                vocab_size=vocab_size,
                num_layers=1,
                hidden_size=hidden_size,
                num_attention_heads=4
            )

        test_layer = masked_lm.MaskedLM(
            embedding_table=xformer_stack.get_embedding_table(),
            output=output
        )
        return test_layer

    def test_layer_creation(self):
        vocab_size = 100
        seq_len = 32
        hidden_size = 64
        num_predictions = 21
        test_layer = self.create_layer(
            vocab_size=vocab_size,
            hidden_size=hidden_size
        )

        lm_input_tensor = tf.keras.Input(shape=(seq_len, hidden_size), batch_size=32)
        masked_positions = tf.keras.Input(shape=(num_predictions,), batch_size=32, dtype=tf.int32)
        output = test_layer(lm_input_tensor, masked_positions)

        expected_output_shape = [32, num_predictions, vocab_size]
        self.assertEqual(expected_output_shape, output.shape.as_list())


if __name__ == '__main__':
    tf.test.main()
