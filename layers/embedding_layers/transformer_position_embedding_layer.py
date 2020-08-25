# -*- coding: utf - 8 -*-

import math
import tensorflow as tf
from layers import utils


class TransformerPositionEmbedding(tf.keras.layers.Layer):
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

        super(TransformerPositionEmbedding, self).__init__(**kwargs)
        self._hidden_size = hidden_size
        self._min_timescale = min_timescale
        self._max_timescale = max_timescale

    def get_config(self):
        config = {
            'hidden_size': self._hidden_size,
            'min_timescale': self._min_timescale,
            'max_timescale': self._max_timescale
        }
        base_config = super(TransformerPositionEmbedding, self).get_config()
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
