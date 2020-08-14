# -*- coding: utf - 8 -*-

import tensorflow as tf

x = tf.random.uniform(minval=0, maxval=100, shape=(4, 4), dtype=tf.int32)
print(x)

# x 的 rank 是 2
# paddings 数组里面需要有 rank=2 个数组
# 每个数组里面有两个数
# 分别是这一维度左右两侧需要填充几行
# 默认情况使用全 0 填充
y = tf.pad(x, paddings=[[1, 0], [0, 2]])
print(y)
