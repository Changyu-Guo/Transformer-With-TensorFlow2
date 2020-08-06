# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from layers import utils


class ClassificationHead(tf.keras.layers.Layer):
    """
    这个 Layer 作用是：给 [CLS] 做分类

    参数说明：
    inner_dim: 对 [CLS] 做一次线性变换，变换到指定维度，以更好的提取特征
    num_classes: 将 [CLS] 分为多少类
    cls_token_idx: 在 sequence 中哪一个 token 是 [CLS]

    该 Layer 接收一个 Tensor 作为输入
    Tensor Shape 应该为 (batch_size, seq_len, hidden_size)
    这应该是 bert encoder 最后一层的输出

    运算流程：
    1. 取出 [CLS] token
    2. 映射到指定维度 (inner dim)
    3. dropout
    4. 映射到 num_classes (分类)

    在不使用 encoder pooler 的时候这个 Layer 可能会有用
    和 encoder pooler 类似，但是在 dense 后加入了 dropout
    如果使用 encoder pooler, 则尽量对 cls_output 进行操作（直接执行分类）

    这里作为一个单独的 Layer 提取出来，在更多的 Model 中可以复用
    """
    def __init__(
            self,
            inner_dim,
            num_classes,
            cls_token_idx=0,
            activation='tanh',
            dropout_rate=0.0,
            initializer='glorot_uniform',
            **kwargs
    ):
        super(ClassificationHead, self).__init__(**kwargs)
        self._dropout_rate = dropout_rate
        self._inner_dim = inner_dim
        self._num_classes = num_classes
        self._activation = tf.keras.activations.get(activation)
        self._initializer = tf.keras.initializers.get(initializer)
        self._cls_token_idx = cls_token_idx

        self.dense = tf.keras.layers.Dense(
            units=inner_dim,
            activation=self._activation,
            kernel_initializer=self._initializer,
            name='pooler_dense'
        )
        self.dropout = tf.keras.layers.Dropout(rate=self._dropout_rate)
        self.output_dense = tf.keras.layers.Dense(
            units=num_classes,
            kernel_initializer=self._initializer,
            name='logits'
        )

    def call(self, features):
        x = features[:, self._cls_token_idx, :]
        x = self.dense(x)
        x = self.dropout(x)
        x = self.output_dense(x)
        return x

    def get_config(self):
        config = {
            'dropout_rate': self._dropout_rate,
            'num_classes': self._num_classes,
            'inner_dim': self._inner_dim,
            'activation': tf.keras.activations.serialize(self._activation),
            'initializer': tf.keras.initializers.serialize(self._initializer)
        }
        config.update(super(ClassificationHead, self).get_config())
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def checkpoint_items(self):
        return {self.dense.name: self.dense}
