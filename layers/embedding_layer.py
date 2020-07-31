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

            # 这个矩阵的作用有两个
            # 1. 作为词典对 word id 做 embedding
            # 2. 将 embedding 从 hidden_size 转回 vocab_size, 最后再用 softmax 转为 logits
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
        根据 mode 的取值不同, Embedding 可以执行两种不同的操作：1. 变过去，2. 变回来

        如果 mode 指定为 embedding，则 inputs 的 shape 必须为 [batch_size, seq_len]
        这个时候会调用 _embedding 方法，按照查表的方式，将每个 seq 中的 id 转为 hidden_size 大小的 embedding

        如果 mode 指定为 linear，则此时应该是在解码器的最后一步，需要将 embedding 从 hidden_size 映射回 vocab_size
        这个时候 inputs 的 shape 必须为 [batch_size, seq_len, hidden_size]
        此时会调用 _linear 方法，将 inputs * shared_weights.T，得到 shape 为 [batch_size, seq_len, vocab_size]
        在此之后会将 softmax 前的 logits 传到外面，由外层代码决定是否要对结果进行 softmax 运算
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
            embeddings = tf.gather(self.share_weights, inputs, axis=0)

            # 因为 pad 也会被 embed 成一组数字
            # 所以需要将本来为 padding 的位置全变为 0

            # (batch_size, seq_len)
            mask = tf.cast(tf.not_equal(inputs, 0), embeddings.dtype)

            # (batch_size, seq_len, 1)
            mask = tf.expand_dims(mask, -1)

            # mask 会广播成 (batch_size, seq_len, hidden_size)
            # 即将 0 的位置广播到整个 hidden_size 大小，实现将整个句子变为 0
            embeddings *= mask

            # 根据 attention is all you need
            # 需要将 embedding 除以 sqrt(d_model)
            embeddings *= self.hidden_size ** 0.5

            return embeddings

    def _linear(self, inputs):
        """
        :param inputs: [batch_size, seq_len, hidden_size]
        :return: [batch_size, seq_len, vocab_size]
        """
        with tf.name_scope('pre_softmax_linear'):
            batch_size = tf.shape(inputs)[0]
            seq_len = tf.shape(inputs)[1]

            # 将所有词展平
            x = tf.reshape(inputs, [-1, self.hidden_size])

            # share_weights 在最后 softmax 之前的一个运算中会进行反向传播
            # 这样就会对 embedding 进行更新
            logits = tf.matmul(x, self.share_weights, transpose_b=True)

            # 将词叠成 seq 再叠成 batch
            return tf.reshape(logits, [batch_size, seq_len, self.vocab_size])
