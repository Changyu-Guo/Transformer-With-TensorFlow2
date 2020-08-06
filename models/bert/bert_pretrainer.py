# -*- coding: utf - 8 -*-

import copy
from typing import List, Optional

import tensorflow as tf
from layers.head_layers.masked_lm import MaskedLM
from networks.classification import Classification


class BertPretrainer(tf.keras.Model):
    def __init__(
            self,
            network,
            num_classes,
            num_token_predictions,
            embedding_table=None,
            activation=None,
            initializer='glorot_uniform',
            output='logits',
            **kwargs
    ):
        super(BertPretrainer, self).__init__()
        self._config = {
            'network': network,
            'num_classes': num_classes,
            'num_token_predictions': num_token_predictions,
            'activation': activation,
            'initializer': initializer,
            'output': output
        }

        self.encoder = network

        # 和 encoder 使用相同的输入
        network_inputs = self.encoder.inputs
        inputs = copy.copy(network_inputs)

        # 得到 encoder 的输出
        # sequence_output: ([layers], batch_size, seq_len, hidden_size)
        # cls_output: (batch_size, hidden_size)
        sequence_output, cls_output = self.encoder(network_inputs)

        if isinstance(sequence_output, list):
            sequence_output = sequence_output[-1]
        if isinstance(cls_output, list):
            cls_output = cls_output[-1]

        # 获取 seq_len
        # 需要预测的 tokens 的数量应该小于 sequence 的总长度
        sequence_output_length = sequence_output.shape.as_list()[1]
        if sequence_output_length is not None and sequence_output_length < num_token_predictions:
            raise ValueError(
                'The passes network\'s output length is %s, which'
                'is less than the requested num_token_predictions %s.'
                % (sequence_output_length, num_token_predictions)
            )

        # 在做 pretrain 的时候还有一个额外输入
        # 就是每个句子需要预测多少个词
        # (batch_size, num_token_predictions)
        masked_lm_positions = tf.keras.Input(
            shape=(num_token_predictions,),
            name='masked_lm_positions',
            dtype=tf.int32
        )
        inputs.append(masked_lm_positions)

        if embedding_table is None:
            embedding_table = self.encoder.get_embedding_table()
        self.masked_lm = MaskedLM(
            embedding_table=embedding_table,
            activation=activation,
            initializer=initializer,
            output=output,
            name='cls/predictions'
        )

        # Masked LM 任务
        # (batch_size * num_masked, vocab_size)
        lm_outputs = self.masked_lm(
            sequence_output,  masked_positions=masked_lm_positions
        )

        # NSP 任务
        self.classification = Classification(
            input_size=cls_output.shape[-1],
            num_classes=num_classes,
            initializer=initializer,
            output=output,
            name='classification'
        )
        sentence_outputs = self.classification(cls_output)

        super(BertPretrainer, self).__init__(
            inputs=inputs,
            outputs=dict(masked_lm=lm_outputs, classification=sentence_outputs),
            **kwargs)

    def get_config(self):
        return self._config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
