# -*- coding: utf - 8 -*-

import tensorflow as tf

x = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
y = tf.tile(x, [2, 3])

initial_log_probs = tf.constant(
    [[0.] + [-float('inf')] * (4 - 1)]
)
alive_log_probs = tf.tile(initial_log_probs, [8, 1])
print(alive_log_probs)