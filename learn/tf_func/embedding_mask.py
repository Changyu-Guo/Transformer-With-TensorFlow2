# -*- coding: utf - 8 -*-

import tensorflow as tf

inputs_ids = tf.random.uniform(
    minval=0, maxval=3, shape=(2, 8), dtype=tf.int32
)

embedding_mask = tf.not_equal(
    inputs_ids, 0
)
embedding_mask = tf.cast(embedding_mask, dtype=tf.float32)

embedded_inputs = tf.random.uniform(
    minval=0, maxval=3, shape=(2, 8, 6),
    dtype=tf.float32
)

print(embedded_inputs * embedding_mask[:, :, None])