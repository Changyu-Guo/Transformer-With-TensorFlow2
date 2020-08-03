# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Classification(tf.keras.Model):
    def __init__(
            self,
            input_size,
            num_classes,
            initializer='glorot_uniform',
            output='logits',
            **kwargs
    ):
        self._self_setattr_tracking = False
        self._config_dict = {
            'input_size': input_size,
            'num_classes': num_classes,
            'initializer': initializer,
            'output': output
        }

        cls_output = tf.keras.Input(
            shape=(input_size,), name='cls_output', dtype=tf.float32
        )
        self.logits = tf.keras.layers.Dense(
            num_classes,
            activation=None,
            kernel_initializer=initializer,
            name='logits'
        )(cls_output)

        policy = tf.keras.mixed_precision.experimental.global_policy()
        if policy.name == 'mixed_bfloat16':
            policy = tf.float32

        predictions = tf.keras.layers.Activation(
            tf.nn.softmax,
            dtype=policy
        )(self.logits)

        if output == 'logits':
            output_tensors = self.logits
        elif output == 'predictions':
            output_tensors = predictions
        else:
            raise ValueError(
                'Unknown output value "%s". output can be either "logits" or '
                '"predictions"' % output
            )
        super(Classification, self).__init__(
            inputs=[cls_output], outputs=[output_tensors], **kwargs
        )

    def get_config(self):
        return self._config_dict

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)