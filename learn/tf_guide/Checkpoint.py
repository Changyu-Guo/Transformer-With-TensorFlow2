# -*- coding: utf - 8 -*-

"""
    Saving a TensorFlow model 通常意味着两件事的其中之一：
    1. Checkpoint
    2. SavedModel

    Checkpoints 抽取模型中的所有参数值(tf.Variable 对象)
    其并不会包含模型的计算图

    SavedModel 不但会保存模型中的所有参数，还会保存模型的计算图
    此时保存下来的模型将不再依赖于源代码
    因此可以通过 TensorFlow Serving、TensorFlow Lite、TensorFlow.js 等进行部署
"""

# 1. Setup
import tensorflow as tf


class Net(tf.keras.Model):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = tf.keras.layers.Dense(5)

    def call(self, x):
        return self.l1(x)


net = Net()

# 2. Saving from tf.keras training APIs
"""
    tf.keras.Model.save_weights 可以保存 checkpoint
    (注意 checkpoint 只包含 weights)
"""
net.save_weights('./tmp/easy_checkpoint')

# 3. Writing checkpoints
"""
    TensorFlow 模型的参数被持久化存储在 tf.Variable 对象中
    tf.Variable 可以直接构建，但通常通过高阶 API 创建
    例如 tf.keras.layers 或 tf.keras.Model
    
    管理 tf.Variable 对象最简单的方法就是使用 Python 的对象
    
    tf.train.Checkpoint、tf.keras.layers.Layer、tf.keras.Model 的子类
    可以自动追踪到它们属性中的 tf.Variable
"""

# 3.1 Manual checkpointing

# 3.1.1 Setup


def toy_dataset():
    inputs = tf.range(10)[:, None]
    labels = inputs * 5 + tf.range(5)[None, :]
    labels = tf.cast(labels, tf.float32)
    return tf.data.Dataset.from_tensor_slices(
        dict(x=inputs, y=labels)
    ).repeat().batch(2)


def train_step(net, example, optimizer):
    with tf.GradientTape() as tape:
        output = net(example['x'])
        loss = tf.reduce_mean(tf.abs(output - example['y']))
    variables = net.trainable_variables
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))
    return loss

# 3.1.2 Create the checkpoint objects


optimizer = tf.keras.optimizers.Adam(0.1)
dataset = toy_dataset()
iterator = iter(dataset)
ckpt = tf.train.Checkpoint(
    step=tf.Variable(1),
    optimizer=optimizer,
    iterator=iterator
)
manager = tf.train.CheckpointManager(ckpt, './tmp/tf_ckpt', max_to_keep=4)

# 3.1.3 Train and checkpoint the model


def train_and_checkpoint(net, manager):
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print('Restored from {}'.format(manager.latest_checkpoint))
    else:
        print('Initializing from scratch')

    for _ in range(50):
        example = next(iterator)
        loss = train_step(net, example, optimizer)
        ckpt.step.assign_add(1)
        if int(ckpt.step) % 10 == 0:
            save_path = manager.save()
            print('Saved checkpoint for step {}: {}'.format(int(ckpt.step), save_path))
            print('loss {:1.2f}'.format(loss.numpy()))


train_and_checkpoint(net, manager)


# 3.1.4 Restore and continue training

optimizer = tf.keras.optimizers.Adam(0.1)
net = Net()
dataset = toy_dataset()
iterator = iter(dataset)
ckpt = tf.train.Checkpoint(
    step=tf.Variable(1),
    optimizer=optimizer,
    net=net,
    iterator=iterator
)
manager = tf.train.CheckpointManager(ckpt, './tmp/tf_ckpt', max_to_keep=3)

train_and_checkpoint(net, manager)

print(manager.checkpoints)
