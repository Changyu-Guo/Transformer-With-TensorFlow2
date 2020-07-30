# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class EmbeddingShareWeights(tf.keras.layers.Layer):
    def __init__(self, vocab_size, hidden_size):
        super(EmbeddingShareWeights, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def build(self, input_shape):
        with tf.name_scope('embedding_and_softmax'):
            # embedding_and_softmax/weights
            self.share_weights = self.add_weight(
                'weights',
                # 每一个 vocab 对应一个向量
                # 向量的大小是 hidden_size
                shape=[self.vocab_size, self.hidden_size],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(
                    mean=0, stddev=self.hidden_size ** -0.5
                )
            )
        super(EmbeddingShareWeights, self).build(input_shape)

    def get_config(self):
        return {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size
        }

    def call(self, inputs, mode='embedding'):
        """
        :param inputs: [batch_size, seq_len] / [batch_size, seq_len, hidden_size]
        :param mode: 可以取 embedding 和 linear
                     如果是 embedding，则将词汇嵌入到 hidden_size [batch_size, seq_len, hidden_size]
                     如果是 linear，则将词汇变换到 vocab_size [batch_size, seq_len, vocab_size]
        """
        if mode == 'embedding':
            return self._embedding(inputs)
        elif mode == 'linear':
            return self._linear(inputs)
        else:
            raise ValueError('model {} is not valid'.format(mode))

    def _embedding(self, inputs):
        """
        :param inputs: [batch_size, seq_len]
        :return: [batch_size, seq_len, hidden_size]
        """
        with tf.name_scope('embedding'):
            # 从 share_weights 中检索出对应词的向量
            # 返回 [batch_size, seq_len, hidden_size]
            embeddings = tf.gather(self.share_weights, inputs)

            # 将本来为 padding 的位置全变为 0
            mask = tf.cast(tf.not_equal(inputs, 0), embeddings.dtype)
            embeddings *= tf.expand_dims(mask, -1)

            # 根据 attention is all you need
            # 需要将 embedding 除以 sqrt(d_model)
            embeddings *= self.hidden_size ** 0.5

            return embeddings

    def _linear(self, inputs):
        """
        :param inputs: [batch_size, seq_len, hidden_size]
        :return: [batch_size, seq_len, vocab_size]
        """
        with tf.name_scope('presoftmax_linear'):
            batch_size = tf.shape(inputs)[0]
            seq_len = tf.shape(inputs)[1]

            x = tf.reshape(inputs, [-1, self.hidden_size])

            # share_weights 在最后 softmax 之前的一个运算中会进行反向传播
            # 这样就会对 embedding 进行更新
            logits = tf.matmul(x, self.share_weights, transpose_b=True)

            return tf.reshape(logits, [batch_size, seq_len, self.vocab_size])