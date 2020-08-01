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
        cls_output = tf.keras.Input(
            shape=(input_size,), name='cls_output', dtype=tf.float32
        )
        self.logits = tf.keras.layers.Dense(
            num_classes,
            activation=None,
            kernel_initializer=initializer,
            name='logits'
        )(cls_output)
        predictions = tf.keras.layers.Activation(tf.nn.softmax)(self.logits)
        if output == 'logits':
            output_tensors = self.logits
        elif output == 'predictions':
            output_tensors = predictions
        else:
            raise ValueError(
                'Unknown output type'
            )
        super(Classification, self).__init__(
            inputs=[cls_output], outputs=[output_tensors], **kwargs
        )