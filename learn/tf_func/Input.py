# -*- coding: utf - 8 -*-

import collections
import tensorflow as tf


class M(tf.keras.Model):
    def __init__(self, units):
        super(M, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(self.units)
        super(M, self).build(input_shape)

    def call(self, inputs):
        inputs_ids = inputs[0]
        targets_ids = inputs[1]
        return self.dense(inputs_ids) + self.dense(targets_ids)


x = tf.keras.Input(shape=(8,), batch_size=None, name='aaa')
xx = tf.keras.Input(shape=(8,), batch_size=None, name='bbb')
m = M(128)
output = m([x, xx])
model = tf.keras.Model([x, xx], output)

x = tf.random.uniform(shape=(2, 8))
xx = tf.random.uniform(shape=(2, 8))

data = collections.OrderedDict()
data['aaa'] = x
data['bbb'] = xx
model(data)
