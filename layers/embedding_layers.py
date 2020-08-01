# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from layers import utils


class WordEmbeddingShareWeights(tf.keras.layers.Layer):
    def __init__(self, vocab_size, hidden_size):
        super(WordEmbeddingShareWeights, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def build(self, input_shape):
        with tf.name_scope('word_embedding_and_softmax'):
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
        super(WordEmbeddingShareWeights, self).build(input_shape)

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


class PositionEmbedding(tf.keras.layers.Layer):
    """BERT"""
    def __init__(
            self,
            initializer='glorot_uniform',
            use_dynamic_slicing=False,
            max_seq_len=None,
            **kwargs
    ):
        if 'dtype' not in kwargs:
            kwargs['dtype'] = tf.float32

        super(PositionEmbedding, self).__init__(**kwargs)

        if use_dynamic_slicing and max_seq_len is None:
            raise ValueError(
                'If use_dynamic_slicing is True, max_seq_len must be set'
            )
        self.max_seq_len = max_seq_len
        self.initializer = tf.keras.initializers.get(initializer)
        self.use_dynamic_slicing = use_dynamic_slicing

    def get_config(self):
        config = {
            'max_seq_len': self.max_seq_len,
            'initializer': tf.keras.initializers.serialize(self.initializer),
            'use_dynamic_slicing': self.use_dynamic_slicing
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        dim_list = input_shape.as_list()

        if len(dim_list) != 3:
            raise ValueError(
                'PositionEmbedding expects a 3-dimensional input tensor '
                'of shape [batch, seq_len, hidden_size]'
            )

        seq_len = dim_list[1]
        hidden_size = dim_list[2]

        if not self.use_dynamic_slicing:
            if seq_len is None:
                raise ValueError(
                    'PositionEmbedding must have use_dynamic_slicing set '
                    'to True (and max_seq_len set) when the '
                    'sequence (1st) dimension of the input is None.'
                )
            if self.max_seq_len is not None:
                raise ValueError(
                    'When use_dynamic_slicing is False, max_seq_len should '
                    "not be specified and we ought to use seq_len to get the "
                    "variable shape."
                )

        if self.max_seq_len is not None:
            weight_seq_len = self.max_seq_len
        else:
            weight_seq_len = seq_len

        self.position_embeddings = self.add_weight(
            'embeddings',
            shape=[weight_seq_len, hidden_size]
        )
        
        super(PositionEmbedding, self).build(input_shape)

    def call(self, inputs):
        input_shape = utils.get_shape_list(inputs, expected_rank=3)
        if self.use_dynamic_slicing:
            position_embeddings = self.position_embeddings[:input_shape[1], :]
        else:
            position_embeddings = self.position_embeddings

        return tf.broadcast_to(position_embeddings, input_shape)


class RelativePositionEmbedding(tf.keras.layers.Layer):
    """Attention is all you need.
        PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    其中 i 表示 hidden_size 的某一维
    在真正的实现中，没有采用 sin 函数和 cos 函数交叉的方式，而是前半段是 sin，后半段是 cos
    另外在实现了通过参数设置调整 timescale
    公式：
        PE(pos, i) = sin/cos(pos * 1 / ((max / min) ^ (i / (d_model // 2 - 1))))
    d_model 除以 2 还要减 1 是因为 下标是从 0 开始的
    对于 1 / (max / min) ^ (i / (d_model // 2 - 1)) 可以变形为
    e ^ (i / (d_model // 2 - 1)) * log ((max / min) -> e ^ (i * log(max / min) / (d_model // 2 - 1))
    将
    """

    def __init__(
            self,
            hidden_size,
            min_timescale=1.0,
            max_timescale=10000.0,
            **kwargs
    ):
        # 其中有些操作需要在 float32 进行
        if 'dtype' not in kwargs:
            kwargs['dtype'] = 'float32'

        super(RelativePositionEmbedding, self).__init__(**kwargs)
        self._hidden_size = hidden_size
        self._min_timescale = min_timescale
        self._max_timescale = max_timescale

    def get_config(self):
        config = {
            'hidden_size': self._hidden_size,
            'min_timescale': self._min_timescale,
            'max_timescale': self._max_timescale
        }
        base_config = super(RelativePositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, length=None):
        """
        :param inputs: [batch_size, seq_len, ...]
        :param length: seq_len
        :return:
        """
        if inputs is None and length is None:
            raise ValueError(
                'If inputs is None, length must be set in'
                'RelativePositionEmbedding()'
            )
        if inputs is not None:
            input_shape = utils.get_shape_list(inputs)
            if length is not None and length != input_shape[1]:
                raise ValueError(
                    'If inputs is not None, length must equal to input_shape[1]'
                )
            length = input_shape[1]
        # pos
        # (seq_len, )
        position = tf.cast(tf.range(length), tf.float32)

        # 将 hidden_size 分为两部分，前半段用 sin 函数处理，后半段用 cos 函数处理
        hidden_index = self._hidden_size // 2

        min_timescale, max_timescale = self._min_timescale, self._max_timescale

        # 计算出常数项
        # 结果是一个常数
        log_timescale_increment = (
                math.log(
                    float(max_timescale) / float(min_timescale)
                ) / (tf.cast(hidden_index, tf.float32) - 1)
        )

        # 加入 hidden index 信息
        # sin 和 cos 的 hidden index 都是从 0 开始
        # (hidden_size / 2, )
        inv_timescales = min_timescale * tf.exp(
            tf.cast(tf.range(hidden_index), tf.float32) *
            -log_timescale_increment
        )

        # 加入 pos 信息
        # (seq_len, 1) * (1, hidden_size / 2) = (seq_len, hidden_size / 2)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)

        # 前半段和后面段连接
        # (seq_len, hidden_size)
        position_embeddings = tf.concat(
            [tf.sin(scaled_time), tf.cos(scaled_time)], axis=1
        )
        return position_embeddings


class OnDeviceEmbedding(tf.keras.layers.Layer):
    def __init__(
            self,
            vocab_size,
            hidden_size,
            initializer='glorot_uniform',
            use_one_hot=False,
            **kwargs
    ):
        super(OnDeviceEmbedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.initializer = initializer
        self.use_one_hot = use_one_hot

    def get_config(self):
        config = {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'initializer': self.initializer,
            'use_one_hot': self.use_one_hot
        }
        base_config = super(OnDeviceEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            'embeddings',
            shape=[self.vocab_size, self.hidden_size],
            initializer=self.initializer,
            dtype=tf.float32
        )
        super(OnDeviceEmbedding, self).build(input_shape)

    def call(self, inputs):
        flat_inputs = tf.reshape(inputs, [-1])
        if self.use_one_hot:
            one_hot_data = tf.one_hot(
                flat_inputs,
                depth=self.vocab_size,
                dtype=self.embeddings.dtype
            )
        else:
            embeddings = tf.gather(self.embeddings, flat_inputs)
        embeddings = tf.reshape(
            embeddings,
            tf.concat([tf.shape(inputs), [self.hidden_size]], axis=0)
        )
        embeddings.set_shape(inputs.shape.as_list() + [self.hidden_size])
        return embeddings
