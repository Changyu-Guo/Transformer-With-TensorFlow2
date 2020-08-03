from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class OnDeviceEmbedding(tf.keras.layers.Layer):
    def __init__(
            self,
            vocab_size,
            embedding_size,
            initializer='glorot_uniform',
            use_one_hot=False,
            use_scale=False,
            **kwargs
    ):
        super(OnDeviceEmbedding, self).__init__(**kwargs)
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._initializer = initializer
        self._use_one_hot = use_one_hot
        self._use_scale = use_scale

    def get_config(self):
        config = {
            'vocab_size': self._vocab_size,
            'embedding_size': self._embedding_size,
            'initializer': self._initializer,
            'use_one_hot': self._use_one_hot,
            'use_scale': self._use_scale
        }
        base_config = super(OnDeviceEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            'embeddings',
            shape=[self._vocab_size, self._embedding_size],
            initializer=self._initializer,
            dtype=tf.float32
        )
        super(OnDeviceEmbedding, self).build(input_shape)

    def call(self, inputs):
        # 将 inputs 展平
        flat_inputs = tf.reshape(inputs, [-1])

        # 先将词转为 one-hot 再嵌入
        if self._use_one_hot:
            one_hot_data = tf.one_hot(
                flat_inputs,
                depth=self._vocab_size,
                dtype=self.embeddings.dtype
            )
            embeddings = tf.matmul(one_hot_data, self.embeddings)
        else:
            embeddings = tf.gather(self.embeddings, flat_inputs)
        embeddings = tf.reshape(
            embeddings,
            tf.concat([tf.shape(inputs), [self._embedding_size]], axis=0)
        )
        embeddings.set_shape(inputs.shape.as_list() + [self._embedding_size])
        if self._use_scale:
            embeddings *= self._embedding_size ** 0.5
        return embeddings
