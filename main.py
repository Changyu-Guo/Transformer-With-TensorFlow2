# -*- coding: utf - 8 -*-

import tensorflow as tf

x = tf.random.uniform(shape=(12, 3))

print(x[:, -1:])
