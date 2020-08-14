# Transformers-With-TensorFlow2 Parameters

## Attention

- `num_attention_heads`
- `size_per_head_for_query_and_key`
- `size_per_head_for_value`
- `attention_dropout_rate`
- `use_bias`
- `output_shape*`
- `attention_axes`
- `return_attention_scores`
- `norm_first`

**注：**

对于`Attention`中的两种`mask`，即`padding_mask`和`look_ahead_mask`，初始情况下一律使用如下`shape`：

`padding_mask`：`(batch_size, seq_len)`

`look_ahead_mask`：`(seq_len, seq_len)`

在`Attention`运算的过程中，再做`shape`上的转变

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

