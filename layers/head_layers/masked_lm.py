# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from layers import utils


class MaskedLM(tf.keras.layers.Layer):
    def __init__(
            self,
            embedding_table,
            activation='relu',
            initializer='glorot_uniform',
            output='logits',
            name='cls/prediction',
            **kwargs
    ):
        super(MaskedLM, self).__init__(name=name, **kwargs)
        self._embedding_table = embedding_table
        self._activation = activation
        self._initializer = tf.keras.initializers.get(initializer)

        if output not in ('predictions', 'logits'):
            raise ValueError(
                'Unknown output value %s. output can be either logits'
                'or predictions' % output
            )
        self._output_type = output

    def build(self, input_shape):
        self._vocab_size, embedding_size = self._embedding_table.shape
        self.dense = tf.keras.layers.Dense(
            units=embedding_size,
            activation=self._activation,
            kernel_initializer=self._initializer,
            name='transformer/dense'
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-12, name='transformer/LayerNorm'
        )
        self.bias = self.add_weight(
            name='output_bias/bias',
            shape=(self._vocab_size,),
            initializer='zeros',
            trainable=True
        )
        super(MaskedLM, self).build(input_shape)

    def call(self, seqs, masked_positions):
        """
        :param seqs: (batch_size, seq_len, hidden_size)
        :param masked_positions: (batch_size, num_masked)
        :return:
        """
        # 获取被 mask 掉的部分的 tensor (encoder output)
        # (batch_size * num_masked, hidden_size)
        masked_tensor = self._gather_indexes(seqs, masked_positions)

        # 从 hidden_size 转回 embedding_size
        # (batch_size * num_masked, embedding_size)
        lm_data = self.dense(masked_tensor)

        # layer norm
        # (batch_size * num_masked, embedding_size)
        lm_data = self.layer_norm(lm_data)

        # 再转回 vocab_size
        # (batch_size * num_masked, vocab_size)
        lm_data = tf.matmul(lm_data, self._embedding_table, transpose_b=True)

        # 加入 bias
        # (batch_size * num_masked, vocab_size)
        logits = tf.nn.bias_add(lm_data, self.bias)

        # 获取被 mask 掉的词的位置
        # (batch_size, num_masked)
        masked_positions_shape = utils.get_shape_list(
            masked_positions, name='masked_positions_tensor'
        )

        # (batch_size, num_masked, vocab_size)
        logits = tf.reshape(
            logits, [-1, masked_positions_shape[1], self._vocab_size]
        )
        if self._output_type == 'logits':
            return logits
        return tf.nn.log_softmax(logits)

    def _gather_indexes(self, seqs_tensor, positions):
        """
        :param seqs_tensor: (batch_size, seq_len, hidden_size)
        :param positions: (batch_size, num_masked)
        :return:
        """
        seqs_shape = utils.get_shape_list(
            seqs_tensor,
            expected_rank=3,
            name='sequence_output_tensor'
        )

        batch_size, seq_len, hidden_size = seqs_shape

        # (batch_size, 1)
        # like [0, 128, ...]
        flat_seqs_offsets = tf.reshape(
            tf.range(0, batch_size, dtype=tf.int32) * seq_len, [-1, 1]
        )

        # (batch_size * num_masked,)
        flat_words_positions = tf.reshape(positions + flat_seqs_offsets, [-1])

        # (batch_size * seq_len, hidden_size)
        flat_words_tensor = tf.reshape(
            seqs_tensor, [batch_size * seq_len, hidden_size]
        )

        # 获取被 mask 掉的 tensor
        # (batch_size * num_masked, hidden_size)
        masked_tensor = tf.gather(flat_words_tensor, flat_words_positions)

        return masked_tensor
