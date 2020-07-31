# -*- coding: utf - 8 -*-

import tensorflow as tf

# 转置
a = tf.reshape(tf.range(6), shape=(2, 3))
print(tf.einsum('ij->ji', a))

# 求和
a = tf.reshape(tf.range(6), shape=(2, 3))
print(tf.einsum('ij->', a))

# 列求和
a = tf.reshape(tf.range(6), shape=(2, 3))
print(tf.einsum('ij->j', a))

# 行求和
a = tf.reshape(tf.range(6), shape=(2, 3))
print(tf.einsum('ij->i', a))

# 矩阵 - 向量 相乘
a = tf.reshape(tf.range(6), shape=(2, 3))
b = tf.range(3)
print(tf.einsum('ij,j->i', a, b))

# 矩阵 - 矩阵 相乘
a = tf.reshape(tf.range(6), shape=(2, 3))
b = tf.reshape(tf.range(15), shape=(3, 5))
print(tf.einsum('ij,jk->ik', a, b))

# 点积
a = tf.range(3)
b = tf.range(3, 6)
print(tf.einsum('i,i->', a, b))

# 矩阵点积
a = tf.reshape(tf.range(6), shape=(2, 3))
b = tf.reshape(tf.range(6, 12), shape=(2, 3))
print(tf.einsum('ij,ij->', a, b))

# 哈达玛积
a = tf.reshape(tf.range(6), shape=(2, 3))
b = tf.reshape(tf.range(6, 12), shape=(2, 3))
print(tf.einsum('ij,ij->ij', a, b))

# 外积
a = tf.range(3)
b = tf.range(3, 7)
print(tf.einsum('i,j->ij', a, b))

# batch 矩阵相乘
a = tf.random.normal(shape=(32, 128, 512))
b = tf.random.normal(shape=(32, 512, 512))
print(tf.einsum('ijk,ikl->ijl', a, b).shape)

# 张量缩约
a = tf.random.normal(shape=(2, 3, 5, 7))
b = tf.random.normal(shape=(11, 13, 3, 17, 5))
print(tf.einsum('abcd,efghi->adefh', a, b).shape)

# 双线性变换
a = tf.random.normal(shape=(2, 3))
b = tf.random.normal(shape=(5, 3, 7))
c = tf.random.normal(shape=(2, 7))
print(tf.einsum('ik,jkl,il->ij', a, b, c))

# 线性变换
a = tf.random.normal(shape=(32, 8))
b = tf.random.normal(shape=(8, 100))
print(tf.einsum('ij,jk->ik', a, b).shape)