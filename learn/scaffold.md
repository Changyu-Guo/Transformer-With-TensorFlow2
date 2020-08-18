# 深度学习模型框架

## 一、数据处理

训练任何模型都需要提前将数据处理成模型可以接收的输入，主要包括如下步骤：

### （一）、数据集获取与加载

第一步首先是将原生数据集加载至内存中，数据集的来源可以是：

1. 直接从本地加载
2. 先从网络下载然后加载至内存，从网络上下载下来的资源要注意是否需要先进行解压缩

### （二）、数据集处理

原生数据一般是以某种规则组织的数据，因此可以按照某种特定的规则循环处理原生数据中的所有数据。

数据集处理最主要的是**构建模型输入数据特征**，以便于对数据集的保存、读取以及输入进模型。

一个典型的数据特征就是机器翻译模型输入数据的特征，包含 `inputs` 和 `targets` 两项，不同的任务需要构建不同的数据特征。

对于自然语言处理，在构建模型输入特征的时候最常见的问题是对句子进行 `tokenize`，然后将 `word` 转为 `ids`，之后根据不同的任务可能还要在 `ids` 后面填充一些 `0` 或是构建 `segment_ids` 和 `inputs_mask` 等

**常见数据特征举例：**

**机器翻译：**

```python
# encode
features = collections.OrderedDict()
features['inputs'] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
features['targets'] = tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# decode
data_fields = {
    'inputs': tf.io.VarLenFeature(tf.int64),
    'targets': tf.io.VarLenFeature(tf.int64)
}
```

### （三）、数据集保存

数据集的保存是指将数据集保存为 `TFRecord` 文件，根据具体任务的不同，需要将数据特征保存为**定长数据**或**变长数据**，另外，还需要根据具体情况决定是否需要将数据集拆分为多个文件保存。

### （四）、数据集读取

从`TFRecord`中读取出数据，并根据具体需要对数据进行 `shuffle`、`padding`、`repeat` 等操作

## 二、模型主体

## 三、模型训练

## 四、推断