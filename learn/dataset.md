# `TensorFlow Dataset`

## `Datasets`



## `API`

### `interleave`

**函数签名：**

```python
interleave(
    map_func,
    cycle_length=None,
    block_length=None,
    num_parallel_calls=None,
    deterministic=None
)
```

**官网描述：**Maps `map_func` across this dataset, and interleaves the results.

使用 `interleave` 的主要目的应该是将多个数据集合并成一个数据集，并且可以控制其按照某种规则**交叉合并**

**交叉合并**举例解释：

假设有 `A` 和 `B` 两个数据集，`A` 数据集中有数据 `1, 2, 3, 4, 5`，`B` 数据集中有 `6, 7, 8, 9, 10`，一种直观的交叉合并的方式就是每次从 `A` 和 `B` 数据集中各取一个以此放入新的数据集 `C` 中，则新的数据集 `C` 最终则为 `1, 6, 2, 7, 3, 8, 4, 9, 5, 10`

交叉合并的意义就是**增大数据随机程度**

`TensorFlow` 给出的 `interleave` 函数则更加的灵活，可以使用参数控制交叉合并的方式，以满足用户的各种需求

**参数说明：**

`map_func`：对 `dataset` 中每一个**元素**应用该函数，需要返回一个 `dataset`

`cycle_length`：每次取多少个 `element` 去调用 `map_func`

`block_length`：依次从每个 `map_func` 返回的 `dataset` 中取多少元素

**官方示例：**

```python
dataset = Dataset.range(1, 6)
dataset = dataset.interleave(
    lambda x: Dataset.from_tensors(x).repeat(6),
    cycle_length=2,
    block_length=4
)
list(dataset.as_numpy_iterator())

打印结果：
[
    1, 1, 1, 1,
    2, 2, 2, 2,
    1, 1,
    2, 2,
    3, 3, 3, 3,
    4, 4, 4, 4,
    3, 3,
    4, 4,
    5, 5, 5, ,5,
    5, 5
]
```

在这个例子中，原始的 `dataset` 中一共有 `5` 个数 `[1, 2, 3, 4, 5]`

由于 `cycle_length` 设置为了 `2`，因此 `interleave` 首先从 `dataset` 中取出两个元素 `1` 和 `2`，并将这两个元素**分别传入** `map_func`，这里是一个匿名函数

经过 `map_func` 处理后返回两个新的`dataset`，分别是 `[1, 1, 1, 1, 1, 1]` 和 `[2, 2, 2, 2, 2, 2]`

接下来会从这两个 `dataset` 中**取元素加入到最终的 `dataset`**，由于 `block_length = 4`，因此首先会从第一个 `dataset` 中取出 `4` 个元素加入到最终 `dataset`，然后再从第二个 `dataset` 中取出 `4` 个元素加入到最终 `dataset`，这样就形成了打印结果中的前两行数据

两个数据集取完第一轮之后，继续进行第二轮读取，由于第二轮读取过程中两个数据集的元素均少于两个，因此**两次均读取了剩余所有元素**，这样就得到了打印结果中的第 `3`、`4` 行

最初得到的两个数据集读取完后，继续将原始 `dataset` 中 `cycle_length` 个元素送入 `map_func` 得到新的数据集，并执行新一轮的操作，以此类推

**同时处理多个文件：**

`interleave` 在实际应用中一个非常重要且常见的用法就是读取多个文件中的数据到一个数据集中，官方示例代码：

```python
filenames = [
    '/var/data/file1.txt',
    '/var/data/file2.txt',
    '/var/data/file3.txt',
    '/var/data/file4.txt'
]
dataset = tf.data.Dataset.from_tensor_slices(filenames)
def parse_fn(filename):
    return tf.data.Dataset.range(10)
dataset = dataset.interleave(
    lambda x: tf.data.TextLineDataset(x).map(parse_fn, num_parallel_calls=1),
    cycle_length=4,
    block_length=16
)
```

在实际情况中，可能会使用如下代码获取所有的文件名：

```python
dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
```

而 `parse_fn` 的内容则可能如下：

```python
def parse_fn(filename):
  return tf.data.TFRecordDataset(filename)
```

