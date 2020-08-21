# -*- coding: utf - 8 -*-

import tensorflow as tf

inputs_ids = tf.keras.Input(name='inputs_ids', shape=(None,), batch_size=None, dtype=tf.int32)
targets_ids = tf.keras.Input(name='targets_ids', shape=(None,), batch_size=None, dtype=tf.int32)


embedding = tf.random.uniform(minval=0, maxval=300, shape=(300, 512), dtype=tf.float32)

inputs = tf.gather(embedding, inputs_ids)
targets = tf.gather(embedding, targets_ids)

dense = tf.keras.layers.Dense(units=128)

inputs = dense(inputs)
targets = dense(targets)

sample_weights = tf.matmul(inputs, targets, transpose_b=True)
attention = tf.matmul(sample_weights, targets)

output_dense = tf.keras.layers.Dense(512)

output = output_dense(attention)

output = tf.matmul(output, embedding, transpose_b=True)

model = tf.keras.Model(
    inputs={
        'inputs_ids': inputs_ids,
        'targets_ids': targets_ids
    },
    outputs=output
)

inputs_ids = tf.random.uniform(minval=0, maxval=300, shape=(2, 128))
targets_ids = tf.random.uniform(minval=0, maxval=300, shape=(2, 127))


model({
    'inputs_ids': inputs_ids,
    'targets_ids': targets_ids
})


class Net(tf.keras.Model):

    def __init__(self):
        super(Net, self).__init__()

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(5)

    def call(self, x):
        return self.dense(x)


net = Net()
print(net(tf.ones(shape=(1, 5))))
