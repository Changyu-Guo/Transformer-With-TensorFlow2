# -*- coding: utf - 8 -*-

import tensorflow as tf

# 按照指定的下标(indices)从指定的数据集(params)的指定维度(axis)抽取数据子集
# 默认情况下 axis=0

# 举个例子，params 是一个二维数组
params = tf.constant([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 0]
])

# 这是在第 0 维度（行维度），抽取出下标为 0 和 1 的行
print(tf.gather(params, [0, 1], axis=0))

# gather 不能访问到 params 中的一个单独值

# 注意 gather 返回的 shape，index 的外层每多一层 []
# 维度就会增加一维，而内部形状跟 params 中的相同
print(tf.gather(params, 0, axis=1))  # shape = (2, )
print(tf.gather(params, [0], axis=1))  # shape = (2, 1)

# 从更高的维度索引值
params = tf.reshape(tf.range(0, 27), shape=(3, 3, 3))

# 索引第一个维度
print(tf.gather(params, [0, 1], axis=0))  # shape = (2, 3, 3)

# 索引第二个维度
print(tf.gather(params, [0, 1], axis=1))  # shape = (3, 2, 3)

# 索引第三个维度
print(tf.gather(params, [0, 1], axis=2))  # shape = (3, 3, 2)

# 也可以分别索引多组
print(tf.gather(params, [[0, 1], [0, 1]], axis=0))  # shape = (2, 2, 3, 3)

# 实战：根据 vocab 将 id 转为 embedding
vocab = tf.random.uniform(shape=(41, 512), minval=0, maxval=40, dtype=tf.int32)

# 转换一个句子
seq = tf.random.uniform(shape=(8,), minval=0, maxval=40, dtype=tf.int32)
print(tf.gather(vocab, seq, axis=0))  # shape = (seq_len, hidden_size)

# 转换一个 batch 的句子
batch_seqs = tf.random.uniform(shape=(32, 8), minval=0, maxval=40, dtype=tf.int32)
print(tf.gather(vocab, batch_seqs, axis=0))  # shape = (batch_size, seq_len, hidden_size)
