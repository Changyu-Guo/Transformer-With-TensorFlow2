# -*- coding: utf - 8 -*-

import tensorflow as tf
from absl import logging
logging.set_verbosity(logging.INFO)


class Net(tf.keras.Model):

    def __init__(self):
        super(Net, self).__init__()
        self.dense = tf.keras.layers.Dense(5)

    def call(self, x):
        return self.dense(x)


def toy_dataset():
    inputs = tf.range(5)[:, None]
    labels = inputs * 5 + tf.range(5)[None, :]
    labels = tf.cast(labels, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(dict(
        x=inputs,
        y=labels
    )).repeat().batch(2)
    return dataset


def train_one_step(model, example, optimizer):
    with tf.GradientTape() as tape:
        output = model(example['x'])
        loss = tf.reduce_mean(tf.square(output - example['y']))
    variables = model.trainable_variables
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))
    return loss


def train(net, optimizer, dataset, ckpt_manager):
    for _ in range(2):
        example = next(dataset)
        print(example['x'])
        loss = train_one_step(net, example, optimizer)

        iterations = optimizer.iterations.numpy()
        ckpt_manager.save()
        logging.info('Save checkpoint for step {}'.format(iterations))
        logging.info('loss {:.2f}'.format(loss))


def main():
    dataset = iter(toy_dataset())
    net = Net()
    optimizer = tf.keras.optimizers.Adam(0.1)
    ckpt = tf.train.Checkpoint(
        net=net,
        optimizer=optimizer,
        dataset=dataset
    )
    ckpt_manager = tf.train.CheckpointManager(
        ckpt,
        directory='./tmp/ckpt_manager',
        max_to_keep=3
    )

    # restore
    if ckpt_manager.latest_checkpoint:
        logging.info('Load checkpoint from {}'.format(ckpt_manager.latest_checkpoint))
        ckpt_manager.restore_or_initialize()
    else:
        logging.info('No checkpoint to load')

    train(net, optimizer, dataset, ckpt_manager)


if __name__ == '__main__':
    main()
