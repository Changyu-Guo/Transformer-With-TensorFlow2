# -*- coding: utf - 8 -*-

import tensorflow as tf

x = tf.random.uniform(shape=(32, 128))
print(tf.expand_dims(x, axis=0))  # shape = (1, 32, 128)
print(tf.expand_dims(x, axis=-1))  # shape = (32, 128, 1)
