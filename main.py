# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf


batch_size = 32
seq_len = 128
hidden_size = 512

x = tf.random.uniform(minval=1, maxval=100, shape=(batch_size, seq_len, hidden_size))
y = tf.random.uniform(minval=1, maxval=100, shape=(batch_size, seq_len, hidden_size))

matmul_times = []
einsum_times = []

for _ in range(10):
    start = time.time()
    for i in range(100000):
        z = tf.matmul(x, y, transpose_b=True)
    end = time.time()
    matmul_times.append(end - start)

    start = time.time()
    for i in range(100000):
        z = tf.einsum('abc,adc->abd', x, y)
    end = time.time()
    einsum_times.append(end - start)

print('matmul: ', tf.reduce_mean(matmul_times).numpy())
print('einsum: ', tf.reduce_mean(einsum_times).numpy())
