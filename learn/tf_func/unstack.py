# -*- coding: utf - 8 -*-

import tensorflow as tf

x = tf.random.uniform(minval=1, maxval=100, shape=(2, 32, 128))
a, b = tf.unstack(x)

print(a)  # shape (32, 128)
