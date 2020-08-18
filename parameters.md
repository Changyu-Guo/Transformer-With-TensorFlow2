# Transformers-With-TensorFlow2 Parameters

## Attention

### `Args`

- `num_attention_heads`
- `size_per_head_for_query_and_key`
- `size_per_head_for_value`
- `attention_dropout_rate`
- `use_bias`
- `output_shape*`
- `attention_axes`
- `return_attention_scores`
- `norm_first`

### `MASK`：

在编码器中，存在一个`inputs`的`self-attention`，该`attention`的`mask`命名为**`inputs_padding_mask`**

在解码器中，存在两个`attention`，分别是`targets`的`self-attention`和编码器输出解码器输入之间的`cross-attention`，这两个`attention`的`mask`分别命名为`targets_combine_mask`和`encoder_decoder_padding_mask`

> 其中`combine_mask`由`targets_look_ahead_mask`和`targets_padding_mask`相加得到

### `MASK Shape`：

对于`Attention`中的两种`mask`，即`padding_mask`和`combine_mask`，初始情况下一律使用如下`shape`：

`padding_mask`：`(batch_size, 1, seq_len)`

`combine_mask`：`(batch_size, seq_len, seq_len)`

在`Attention`运算的过程中，再做`shape`上的转变

### `MASK` 生成样例：

**`padding mask`**

```python
def get_padding_mask(seqs, padding_value=0):
    return tf.cast(tf.math.equal(seqs, 0), tf.float32)[:, tf.newaxis, :]
```

**`look_ahead_mask`**

```python
def get_look_ahead_mask(seq_len):
    return 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
```

**`combine mask`**

```python
def get_combine_mask(seqs, padding_value):
    return tf.maximum(
        get_padding_mask(seqs, padding_value),
        get_look_ahead_mask(tf.shape(seqs)[1])
    )
```

> 在 `Attention Mask` 中，`True / 1` 表示是`PAD`，`False / 0` 表示不是`PAD`

而在`softmax`时应该进行如下运算：

```python
if mask is not None:
    # 一般而言，scores 的 shape 为：(batch_size, num_heads, seq_len, seq_len)
    # 而 mask 的 shape 则为 (batch_size, seq_len) 或 (batch_size, seq_len, seq_len)
    # 所以一般会在 axis=1 这一维度对 mask 进行维度扩展
    for _ in range(len(scores.shape) - len(mask.shape)):
        mask = tf.expand_dims(mask, axis=self.mask_expansion_axes)
    
    # 将是 PAD (值为 1) 的地方变为非常大的负数
    # 而不是 PAD 的地方 (值为 0 ) 保持不变
    adder = tf.cast(mask, scores.dtype) * -10000.0
    scores += adder
```



## Position-Wise Feed-Forward Network

- `intermediate_size`
- `intermediate_activation`
- `hidden_dropout_rate`

## Common

- `kernel_initializer`
- `bias_initializer`
- `kernel_regularizer`
- `bias_regularizer`
- `kernel_constraint`
- `bias_constraint`
- `activity_regularizer`

## Other

- `hidden_dropout_rate`

## Bert Config

- `attention_probs_dropout_prob`
- `hidden_act`
- `hidden_dropout_prob`
- `hidden_size`
- `initializer_range`
- `intermediate_size`
- `max_position_embeddings`
- `num_attention_heads`
- `num_hidden_layers`
- `type_vocab_size`
- `vocab_size`

## `Tensor Flow`

### `Transformer`

#### `train:`

#### `predict:`