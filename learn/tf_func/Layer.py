# -*- coding: utf - 8 -*-

import tensorflow as tf


class SimpleDense(tf.keras.layers.Layer):

    def __init__(self, units=32):
        super(SimpleDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name='w',
            shape=(input_shape[-1], self.units),
            dtype=tf.float32,
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='b',
            shape=(self.units,),
            dtype=tf.float32,
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


linear_layer = SimpleDense(4)

y = linear_layer(tf.ones(shape=(3, 3)))
assert len(linear_layer.weights) == 2
assert len(linear_layer.trainable_weights) == 2