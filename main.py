# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

batch_size = 2
seq_len_query = 4
seq_len_key_value = 6
hidden_size = 8


query = tf.random.uniform(
    minval=0,
    maxval=100,
    shape=(batch_size, seq_len_query, hidden_size)
)

key = tf.random.uniform(
    minval=0,
    maxval=100,
    shape=(batch_size, seq_len_key_value, hidden_size)
)

value = tf.random.uniform(
    minval=0,
    maxval=100,
    shape=(batch_size, seq_len_key_value, hidden_size)
)

# (batch_size, seq_len_query, seq_len_key_value)
attention = tf.matmul(query, key, transpose_b=True)

mask = tf.random.uniform(
    minval=0,
    maxval=1,
    shape=(batch_size, seq_len_query, seq_len_key_value)
)

print(attention + mask)
