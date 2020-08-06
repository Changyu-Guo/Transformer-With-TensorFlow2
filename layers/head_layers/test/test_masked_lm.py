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

        lm_input_tensor = tf.keras.Input(shape=(seq_len, hidden_size))
        masked_positions = tf.keras.Input(shape=(num_predictions,), dtype=tf.int32)
        output = test_layer(lm_input_tensor, masked_positions)

        expected_output_shape = [None, num_predictions, vocab_size]
        self.assertEqual(expected_output_shape, output.shape.as_list())

    def test_layer_invocation_with_external_logits(self):
        vocab_size = 100
        sequence_length = 32
        hidden_size = 64
        num_predictions = 21
        xformer_stack = bert_encoder.BertEncoder(
            vocab_size=vocab_size,
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
        )
        test_layer = self.create_layer(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            xformer_stack=xformer_stack,
            output='predictions'
        )
        logit_layer = self.create_layer(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            xformer_stack=xformer_stack,
            output='logits'
        )

        lm_input_tensor = tf.keras.Input(shape=(sequence_length, hidden_size))
        masked_positions = tf.keras.Input(shape=(num_predictions,), dtype=tf.int32)
        output = test_layer(lm_input_tensor, masked_positions)
        logit_output = logit_layer(lm_input_tensor, masked_positions)
        logit_output = tf.keras.layers.Activation(tf.nn.log_softmax)(logit_output)
        logit_layer.set_weights(test_layer.get_weights())
        model = tf.keras.Model([lm_input_tensor, masked_positions], output)
        logits_model = tf.keras.Model(
            [lm_input_tensor, masked_positions], logit_output
        )

        batch_size = 3
        lm_input_data = 10 * np.random.random_sample(
            (batch_size, sequence_length, hidden_size))
        masked_position_data = np.random.randint(
            sequence_length, size=(batch_size, num_predictions))

        ref_outputs = model([lm_input_data, masked_position_data])
        outputs = logits_model([lm_input_data, masked_position_data])

        expected_output_shape = (batch_size, num_predictions, vocab_size)
        self.assertEqual(expected_output_shape, ref_outputs.shape)
        self.assertEqual(expected_output_shape, outputs.shape)
        self.assertAllClose(ref_outputs, outputs)

    def test_layer_invocation(self):
        vocab_size = 100
        sequence_length = 32
        hidden_size = 64
        num_predictions = 21
        test_layer = self.create_layer(
            vocab_size=vocab_size,
            hidden_size=hidden_size
        )

        lm_input_tensor = tf.keras.Input(shape=(sequence_length, hidden_size))
        masked_positions = tf.keras.Input(shape=(num_predictions,), dtype=tf.int32)
        output = test_layer(lm_input_tensor, masked_positions)
        model = tf.keras.Model([lm_input_tensor, masked_positions], output)

        batch_size = 3
        lm_input_data = 10 * np.random.random_sample(
            (batch_size, sequence_length, hidden_size))
        masked_position_data = np.random.randint(
            2, size=(batch_size, num_predictions))
        _ = model.predict([lm_input_data, masked_position_data])

    def test_unknown_output_type_fails(self):
        with self.assertRaisesRegex(ValueError, 'Unknown output value "bad".*'):
            _ = self.create_layer(
                vocab_size=8, hidden_size=8, output='bad')


if __name__ == '__main__':
    tf.test.main()
