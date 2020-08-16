# -*- coding: utf - 8 -*-

import tensorflow as tf

x = tf.random.uniform(shape=(64, 100))

try:
    y = tf.squeeze(input=x, axis=[-1])
except tf.errors.InvalidArgumentError:
    print('被 squeeze 的 dim 只能是 1')
