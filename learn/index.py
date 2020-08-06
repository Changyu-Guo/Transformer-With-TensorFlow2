# -*- coding: utf - 8 -*-

import tensorflow as tf

x = tf.random.uniform(minval=0, maxval=100, shape=(32, 512), dtype=tf.int32)

print(x[:, 0, :])  # error
