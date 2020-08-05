# -*- coding: utf - 8 -*-

import tensorflow as tf
from networks.span_labeling import SpanLabeling


class BertSpanLabeler(tf.keras.Model):
    def __init__(
            self,
            network,
            initializer='glorot_uniform',
            output='logits',
            **kwargs
    ):
        self._network = network
        self._config = {
            'network': network,
            'initializer': initializer,
            'output': output
        }

        # 使用原始 encoder 的输入作为当前模型的输入
        inputs = network.inputs

        # 调用模型
        # 只从结果中取出最后一个的结果
        # (batch_size, seq_len, hidden_size)
        sequence_output, _ = network(inputs)

        # 对结果进行处理
        self.span_labeling = SpanLabeling(
            input_size=sequence_output.shape[-1],
            initializer=initializer,
            output=output,
            name='span_labeling'
        )
        start_logits, end_logits = self.span_labeling(sequence_output)
        start_logits = tf.keras.layers.Lambda(
            tf.identity, name='start_positions'
        )(start_logits)
        end_logits = tf.keras.layers.Lambda(
            tf.identity, name='end_positions'
        )(end_logits)

        logits = [start_logits, end_logits]

        super(BertSpanLabeler, self).__init__(
            inputs=inputs, outputs=logits, **kwargs
        )

    @property
    def checkpoint_items(self):
        return dict(encoder=self._network)

    def get_config(self):
        return self._config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
