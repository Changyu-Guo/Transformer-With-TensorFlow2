# -*- coding: utf - 8 -*-

import tensorflow as tf

x = tf.reshape(
    tf.range(0, 32, dtype=tf.int32) * 128, [-1, 1]
)

y = tf.random.uniform(minval=1, maxval=100, shape=(32, 6), dtype=tf.int32)

print(tf.reshape(x + y, (-1,)))
