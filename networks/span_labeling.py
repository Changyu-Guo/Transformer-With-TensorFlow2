# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class SpanLabeling(tf.keras.Model):
    def __init__(
            self,
            input_size,
            activation=None,
            initializer='glorot_uniform',
            output='logits',
            **kwargs
    ):
        self._config = {
            'input_size': input_size,
            'activation': activation,
            'initializer': initializer,
            'output': output
        }

        # (batch_size, seq_len, hidden_size)
        sequence_data = tf.keras.layers.Input(
            shape=(None, input_size),
            dtype=tf.float32,
            name='sequence_data'
        )

        # 将最后一层的输出映射为两个数字
        # 这样一个 sequence 表示开始 logits
        # 另一个 sequence 表示结束 logits
        # (batch_size, seq_len, 2)
        intermediate_logits = tf.keras.layers.Dense(
            2,  # start position and end position
            activation=activation,
            kernel_initializer=initializer,
            name='predictions/transform/logits'
        )(sequence_data)

        self.start_logits, self.end_logits = tf.keras.layers.Lambda(
            self._split_output_tensor
        )(intermediate_logits)

        start_predictions = tf.keras.layers.Activation(
            tf.nn.log_softmax
        )(self.start_logits)
        end_predictions = tf.keras.layers.Activation(
            tf.nn.log_softmax
        )(self.end_logits)

        if output == 'logits':
            output_tensors = [self.start_logits, end_predictions]
        elif output == 'predictions':
            output_tensors = [start_predictions, end_predictions]
        else:
            raise ValueError(
                'Unknown output value %s. output can be either logits'
                'or predictions' % output
            )

        super(SpanLabeling, self).__init__(
            inputs=[sequence_data], outputs=output_tensors, **kwargs
        )

    def _split_output_tensor(self, tensor):
        # 转为 (2, batch_size, seq_len)
        transposed_tensor = tf.transpose(tensor, [2, 0, 1])
        # 变为两个 (batch_size, seq_len)
        return tf.unstack(transposed_tensor)

    def get_config(self):
        return self._config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
