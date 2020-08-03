# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class MaskedSoftmax(tf.keras.layers.Layer):
    def __init__(self, mask_expansion_axes=None, normalization_axes=None, **kwargs):
        self._mask_expansion_axes = mask_expansion_axes
        # 默认只标准化最后一层
        if normalization_axes is None:
            self._normalization_axes = (-1,)
        else:
            self._normalization_axes = normalization_axes
        super(MaskedSoftmax, self).__init__(**kwargs)

    def call(self, scores, mask=None):
        """
        :param scores: [batch_size, num_heads, seq_len, seq_len]
        :param mask: [1, 1, seq_len, seq_len]
        :return:
        """
        if mask is not None:
            for _ in range(len(scores.shape) - len(mask.shape)):
                mask = tf.expand_dims(mask, axis=self._mask_expansion_axes)

            # 1 代表没有被 mask
            # 0 代表被 mask
            # 将 mask 的部分，即值为 0 的部分变为非常大的负数
            adder = (1.0 - tf.cast(mask, scores.dtype)) * -10000.0
            scores += adder

        # softmax
        if len(self._normalization_axes) == 1:
            return tf.nn.softmax(scores, axis=self._normalization_axes[0])
        else:
            return tf.math.exp(scores - tf.math.reduce_logsumexp(
                scores, axis=self._normalization_axes, keepdims=True
            ))

    def get_config(self):
        config = {
            'mask_expansion_axes': self._mask_expansion_axes,
            'normalization_axes': self._normalization_axes
        }
        base_config = super(MaskedSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))