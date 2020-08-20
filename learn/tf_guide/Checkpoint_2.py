# -*- coding: utf - 8 -*-

import tensorflow as tf
from absl import logging
logging.set_verbosity(logging.INFO)


class Net(tf.keras.Model):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = tf.keras.layers.Dense(5)

    def call(self, x):
        return self.l1(x)


def toy_dataset():
    inputs = tf.range(10)[:, None]
    labels = inputs * 5 + tf.range(5)[None, :]
    labels = tf.cast(labels, tf.float32)
    return tf.data.Dataset.from_tensor_slices(
        dict(x=inputs, y=labels)
    ).repeat().batch(2)


def train_one_step(net, example, optimizer):
    with tf.GradientTape() as tape:
        output = net(example['x'])
        loss = tf.reduce_mean(tf.abs(output - example['y']))
    variables = net.trainable_variables
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))
    return loss


def train_and_checkpoint(net, manager, optimizer, dataset):
    if manager.latest_checkpoint:
        logging.info('Load checkpoint from {}'.format(manager.latest_checkpoint))
        manager.restore_or_initialize()
    else:
        logging.info('Initializing')

    for _ in range(2):
        example = next(dataset)
        loss = train_one_step(net, example, optimizer)
        iterations = optimizer.iterations.numpy()
        if iterations % 1 == 0:
            save_path = manager.save()
            print('Saved checkpoint for step {}: {}'.format(iterations, save_path))
            print('loss {:1.2f}'.format(loss.numpy()))


net = Net()
optimizer = tf.keras.optimizers.Adam(0.1)
save_path = './tmp/tf_ckpts'
ckpt = tf.train.Checkpoint(
    net=net,
    optimizer=optimizer,
    data_iter=iter(toy_dataset())
)
manager = tf.train.CheckpointManager(
    checkpoint=ckpt,
    directory=save_path,
    max_to_keep=3,
    step_counter=optimizer.iterations
)
dataset = iter(toy_dataset())
train_and_checkpoint(net, manager, optimizer, dataset)
