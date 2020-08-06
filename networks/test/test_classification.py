# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow.python.keras import keras_parameterized
from networks import classification


class ClassificationTest(keras_parameterized.TestCase):
    @parameterized.parameters(1, 10)
    def test_network_creation(self, num_classes):
        input_size = 512
        test_object = classification.Classification(
            input_size=input_size, num_classes=num_classes
        )
        cls_data = tf.keras.Input(shape=(input_size,), dtype=tf.float32)
        output = test_object(cls_data)

        expected_output_shape = shape = [None, num_classes]
        self.assertAllEqual(expected_output_shape, output.shape.as_list())

    @parameterized.parameters(1, 10)
    def test_network_invocation(self, num_classes):
        input_size = 512
        test_object = classification.Classification(
            input_size=input_size, num_classes=num_classes, output='predictions')

        cls_data = tf.keras.Input(shape=(input_size,), dtype=tf.float32)
        output = test_object(cls_data)

        model = tf.keras.Model(cls_data, output)
        input_data = 10 * np.random.random_sample((3, input_size))
        _ = model.predict(input_data)

    def test_network_invocation_with_internal_logits(self):

        input_size = 512
        num_classes = 10
        test_object = classification.Classification(
            input_size=input_size, num_classes=num_classes, output='predictions')

        cls_data = tf.keras.Input(shape=(input_size,), dtype=tf.float32)
        output = test_object(cls_data)
        model = tf.keras.Model(cls_data, output)
        logits_model = tf.keras.Model(test_object.inputs, test_object.logits)

        batch_size = 3
        input_data = 10 * np.random.random_sample((batch_size, input_size))
        outputs = model.predict(input_data)
        logits = logits_model.predict(input_data)

        expected_output_shape = (batch_size, num_classes)
        self.assertEqual(expected_output_shape, outputs.shape)
        self.assertEqual(expected_output_shape, logits.shape)

        input_tensor = tf.keras.Input(expected_output_shape[1:])
        output_tensor = tf.keras.layers.Activation(tf.nn.log_softmax)(input_tensor)
        softmax_model = tf.keras.Model(input_tensor, output_tensor)

        calculated_softmax = softmax_model.predict(logits)
        self.assertAllClose(outputs, calculated_softmax)

    @parameterized.parameters(1, 10)
    def test_network_invocation_with_internal_and_external_logits(
            self, num_classes
    ):
        input_size = 512
        test_object = classification.Classification(
            input_size=input_size, num_classes=num_classes, output='logits'
        )

        cls_data = tf.keras.Input(shape=(input_size,), dtype=tf.float32)
        output = test_object(cls_data)
        model = tf.keras.Model(cls_data, output)
        logits_model = tf.keras.Model(test_object.inputs, test_object.logits)

        batch_size = 3
        input_data = 10 * np.random.random_sample((batch_size, input_size))
        outputs = model.predict(input_data)
        logits = logits_model.predict(input_data)

        expected_output_shape = (batch_size, num_classes)
        self.assertEqual(expected_output_shape, outputs.shape)
        self.assertEqual(expected_output_shape, logits.shape)

        self.assertAllClose(outputs, logits)

    def test_network_invocation_with_logit_output(self):
        input_size = 512
        num_classes = 10
        test_object = classification.Classification(
            input_size=input_size, num_classes=num_classes, output='predictions')
        logit_object = classification.Classification(
            input_size=input_size, num_classes=num_classes, output='logits')
        logit_object.set_weights(test_object.get_weights())

        cls_data = tf.keras.Input(shape=(input_size,), dtype=tf.float32)
        output = test_object(cls_data)
        logit_output = logit_object(cls_data)

        model = tf.keras.Model(cls_data, output)
        logits_model = tf.keras.Model(cls_data, logit_output)

        batch_size = 3
        input_data = 10 * np.random.random_sample((batch_size, input_size))
        outputs = model.predict(input_data)
        logits = logits_model.predict(input_data)

        expected_output_shape = (batch_size, num_classes)
        self.assertEqual(expected_output_shape, outputs.shape)
        self.assertEqual(expected_output_shape, logits.shape)

        input_tensor = tf.keras.Input(expected_output_shape[1:])
        output_tensor = tf.keras.layers.Activation(tf.nn.log_softmax)(input_tensor)
        softmax_model = tf.keras.Model(input_tensor, output_tensor)

        calculated_softmax = softmax_model.predict(logits)
        self.assertAllClose(outputs, calculated_softmax)

    def test_serialize_deserialize(self):
        network = classification.Classification(
            input_size=128,
            num_classes=10,
            initializer='zeros',
            output='predictions'
        )

        new_network = classification.Classification.from_config(
            network.get_config())

        _ = new_network.to_json()

        self.assertAllEqual(network.get_config(), new_network.get_config())

    def test_unknown_output_type_fails(self):
        with self.assertRaisesRegex(ValueError, 'Unknown output value "bad".*'):
            _ = classification.Classification(
                input_size=128, num_classes=10, output='bad'
            )


if __name__ == '__main__':
    tf.test.main()