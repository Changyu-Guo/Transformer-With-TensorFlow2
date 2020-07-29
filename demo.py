# -*- coding: utf - 8 -*-

import tensorflow as tf

a = tf.random.normal(shape=(50, 128))
print(tf.gather(a, [[1, 2], [3, 4]]).shape)