# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf


class Classification(tf.keras.Model):
    
    def __init__(self, hidden_size, num_classes, **kwargs):
        self._hidden_size = hidden_size
        self._num_classes = num_classes
        
        super(Classification, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense_layer = tf.keras.layers.Dense(
            units=self._hidden_size,
            activation='relu'
        )
        self.output_dense = tf.keras.layers.Dense(
            units=self._num_classes,
        )

    def call(self, inputs):
        return self.output_dense(self.dense_layer(inputs))


x = tf.keras.Input(shape=(100,), name='abc')
internal_model = Classification(256, 10)
y = internal_model(x)
model = tf.keras.Model(inputs=x, outputs=y)

x = tf.random.uniform(minval=0, maxval=100, shape=(32, 100))
y = model(inputs={
    'abc': x
})

print(y)