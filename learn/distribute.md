# `TensorFlow Distribute Train`

## `MirroredStrategy`

### 原理

`MirroredStrategy` 分布式训练步骤：

1. 在训练开始之前，在所有 `N` 个计算设备上均复制一份完整的模型
2. 每次训练传入一个批次的数据时，将数据平均分为 `N` 份，分别传入 `N` 个计算设备
3. `N` 个计算设备使用本地变量（镜像变量）分别计算自己所获得的部分数据和梯度
4. 使用分布式计算的 `All-Reduce` 操作，在计算设备间高效交换梯度数据并进行求和，使得最终每个设备都有了所有设备的梯度和
5. 使用梯度求和的结果更新本地变量（镜像变量）
6. 当所有设备均更新本地变量后，进行下一轮训练（因此该并行策略是同步的）

> 默认情况下，`TensorFlow`中的`MirroredStrategy`策略使用 `NVIDIA NCCL` 进行 `All-Reduce` 操作

### `fit`

如果使用 `model.fit` 进行模型训练，则只需要在 `strategy` 的 `scope` 下创建模型并编译模型即可，其他部分代码均保持不变（**不需要考虑数据，因为在 `model.fit` 的内部会进行处理**）

代码：

```python
# 定义 strategy
strategy = tf.distribute.MirroredStrategy()

# 在 strategy scope 下定义并编译模型
with strategy.scope():
    model = create_model(params)
    model.compile(
        optimizer,
        loss,
        metrics,
        ...
    )

# fit 不需要在 strategy scope 内执行
model.fit()
```

### `custom train loop`

如果使用`custom train loop`，则需要如下步骤完成分布式训练：

1. 创建 `strategy`
2. 在 `strategy scope`内部，创建 `model` 和 `optimizer`
3. 基于创建的 `strategy`，将数据集变成可分布式读取的
4. 定义一个训练一个 `step` 的函数
5. 使用`strategy.run(one_step_func, args=(one_batch_data,))` 来调用训练一个 `step` 的函数

> 注意，在第 `5` 步中，一个 `batch` 的 `data` 会被平均分配到不同的设备中进行训练

在一般情况下，可以使用如下样例代码：

```python
mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    model = create_model()
    optimizer = tf.keras.optimizers.SGD()
    
@tf.function
def train_steps(iterator, steps):
    
    def _step_fn(inputs):
        inputs, targets = inputs
        with tf.GradientTape() as tape:
            logits = model(...)
            loss = loss_func(logits, targets)
            scaled_loss = loss / self.distribution_strategy.num_replicas_in_sync
        tvars = list({id(v): v for v in model.trainable_variables}.values())
        grads = tape.gradient(scaled_loss, tvars)
        optimizer.apply_gradients(zip(grads, tvars))
        train_loss_metric.update_state(loss)
            
    for _ in tf.range(steps):
        train_loss_metric.reset_states()
        self.distribution_strategy.run(
            _step_fn, args=(next(data_iter),)
        )
```

使用这段代码，只需要给 `train_steps` 传入整个数据集和一个 `steps`，其中 `steps` 可根据 `checkpoint` 和 `summary` 的周期或者是`evaluate`的周期而定

> 注意此处的每个 `batch` 也会被平均分配到每个设备上进行训练，因此要合理设置 `batch size`
>
> **这个代码段同样适用于 `TPU` 分布式训练**