# -*- coding: utf - 8 -*-

import tensorflow as tf


class Dense(tf.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            name=None
    ):
        super(Dense, self).__init__(name=name)
        self.w = tf.Variable(
            tf.random.normal([input_dim, output_dim], name='w')
        )
        self.b = tf.Variable(tf.zeros([output_dim]), name='b')

    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return y


dense = Dense(10, 100, 'dense')
print(dense.trainable_variables)

tf.keras.Model