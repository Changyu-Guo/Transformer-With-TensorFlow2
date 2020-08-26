# -*- coding: utf - 8 -*-

import math
import string
import collections
import numpy as np
import tensorflow as tf
from layers.attention_layers.einsum_dense import EinsumDense
from layers.attention_layers import masked_softmax_layer

_CHR_IDX = string.ascii_lowercase


def _build_attention_equation(rank, attention_axes):
    """
    :param rank: 4
    :param attention_axes:
    :return:
    """
    # Query 矩阵维度 (abcd)
    query_notation = _CHR_IDX[:rank]
    # num_heads 也算是 batch dims (在运算过程中可以随意变换位置)
    batch_dims = tuple(np.delete(range(rank), attention_axes + (rank - 1,)))
    letter_offset = rank

    # Key or Value 矩阵维度
    # Key or Value 矩阵和 Query 矩阵在 batch dims 上保持一致
    # 另外，Key 矩阵最后一维和 Query 矩阵相同 (即 hidden_size 相同)
    key_or_value_notation = ""
    for i in range(rank):
        if i in batch_dims or i == rank - 1:
            key_or_value_notation += query_notation[i]
        else:
            key_or_value_notation += _CHR_IDX[letter_offset]
            letter_offset += 1

    product_notation = "".join(
        # dot-product 之后 batch_size 和 num_heads 变换到了前面
        # 所有在这里先添加 Query 矩阵的 batch dims
        [query_notation[i] for i in batch_dims] +

        # 添加 Query 矩阵的 attention axes
        [query_notation[i] for i in attention_axes] +

        # 添加 Key 矩阵的 attention axes
        [key_or_value_notation[i] for i in attention_axes]
    )

    # Query * Key: abcd,aecd->acbe
    dot_product_equation = "%s,%s->%s" % (
        query_notation, key_or_value_notation, product_notation
    )

    attn_scores_rank = len(product_notation)

    # 乘以 Value 矩阵，变回 Query 维度
    # 1. num_heads 变换回倒数第二维度
    # 2. seq_len 变为 Query 的第二维度
    # 3. hidden_size 变为 Value 的 hidden_size
    # 4. batch_size 保持不变
    # acbe,aecd->abcd
    combine_equation = "%s,%s->%s" % (
        product_notation, key_or_value_notation, query_notation
    )

    return dot_product_equation, combine_equation, attn_scores_rank


def _build_projection_equation(pin_dims, bound_dims, output_dims):
    """
    :param pin_dims: batch_size 和 seq_len
    :param bound_dims: hidden_size
    :param output_dims: num_heads 和 size_per_head
    :return: 经过变换之后会升一个维度，因为是多头注意力机制
    """
    input_str = ''
    kernel_str = ''
    output_str = ''
    bias_axes = ''

    # pin_dims 在变换的过程中被保留下来
    letter_offset = 0
    for i in range(pin_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        output_str += char

    # bound_dims 在在变换的过程中被抵消掉
    # 只存在于 input_str 的末尾和 kernel_str 的开头
    letter_offset += pin_dims
    for i in range(bound_dims):
        char = _CHR_IDX[i + letter_offset]
        input_str += char
        kernel_str += char

    # output_dims 为新的维度
    # 新的维度存在于 kernel_str 的末尾以及 output_str 的末尾
    letter_offset += bound_dims
    for i in range(output_dims):
        char = _CHR_IDX[i + letter_offset]
        kernel_str += char
        output_str += char
        bias_axes += char

    # abc,cde->abde
    # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, num_heads, size_per_head]
    equation = '%s,%s->%s' % (input_str, kernel_str, output_str)

    return equation, bias_axes, len(output_str)


def _get_output_shape(output_rank, known_last_dims):
    # 该输入可以帮助 EinsumDense 确定 kernel 的形状
    # bound_dims 的维度填写为 None，由内部自己推算得出
    # 后续的维度则一定要自己指定
    return [None] * (output_rank - len(known_last_dims)) + list(known_last_dims)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(
            self,
            num_attention_heads,
            size_per_head_for_query_and_key,
            size_per_head_for_value=None,
            attention_dropout_rate=0.0,
            use_bias=True,
            output_shape=None,
            attention_axes=None,
            return_attention_scores=False,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self._num_attention_heads = num_attention_heads
        self._size_per_head_for_query_and_key = size_per_head_for_query_and_key
        self._size_per_head_for_value = size_per_head_for_value if size_per_head_for_value else \
            size_per_head_for_query_and_key
        self._attention_dropout_rate = attention_dropout_rate
        self._use_bias = use_bias
        self._output_shape = output_shape
        self._return_attention_scores = return_attention_scores
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)
        if attention_axes is not None and not isinstance(attention_axes, collections.abc.Sized):
            self._attention_axes = (attention_axes,)
        else:
            self._attention_axes = attention_axes
        self._built_from_signature = False

    def _build_from_signature(self, query, value, key=None):
        """
        :param query: query Tensor or TensorShape
        :param value: value Tensor or TensorShape
        :param key: key Tensor or TensorShape or None
        :return:
        """
        self._built_from_signature = True

        if hasattr(query, 'shape'):
            query_shape = tf.TensorShape(query.shape)
        else:
            query_shape = query

        if hasattr(value, 'shape'):
            value_shape = tf.TensorShape(value.shape)
        else:
            value_shape = value

        if key is None:
            key_shape = value_shape
        elif hasattr(key, 'shape'):
            key_shape = tf.TensorShape(key.shape)
        else:
            key_shape = key

        common_kwargs = dict(
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint
        )

        with tf.init_scope():
            # 默认情况下只在 hidden_size 那个维度作变换
            # 因此前两个维度固定不变，pin_dims 为 query/key/value's shape - 1
            # 而剩下的 hidden_size 为边缘维度，用来做变换
            # 由于变换后需要将 hidden_size 切分为 (num_heads, key_size)
            # 因此 output_dims 为 2
            einsum_equation, bias_axes, output_rank = _build_projection_equation(
                query_shape.rank - 1, bound_dims=1, output_dims=2
            )
            self._query_dense = EinsumDense(
                einsum_equation,
                output_shape=_get_output_shape(
                    output_rank - 1,  # -1 是因为 batch_size
                    [self._num_attention_heads, self._size_per_head_for_query_and_key]
                ),
                bias_axes=bias_axes if self._use_bias else None,
                name='query',
                **common_kwargs
            )

            # 构造 key dense
            einsum_equation, bias_axes, output_rank = _build_projection_equation(
                key_shape.rank - 1, bound_dims=1, output_dims=2
            )
            self._key_dense = EinsumDense(
                einsum_equation,
                output_shape=_get_output_shape(
                    output_rank - 1,
                    [self._num_attention_heads, self._size_per_head_for_query_and_key]
                ),
                bias_axes=bias_axes if self._use_bias else None,
                name="key",
                **common_kwargs
            )

            # 构造 value dense
            einsum_equation, bias_axes, output_rank = _build_projection_equation(
                value_shape.rank - 1,
                bound_dims=1,
                output_dims=2
            )
            self._value_dense = EinsumDense(
                einsum_equation,
                output_shape=_get_output_shape(
                    output_rank - 1,
                    [self._num_attention_heads, self._size_per_head_for_value]
                ),
                bias_axes=bias_axes if self._use_bias else None,
                name="value",
                **common_kwargs
            )

            self.build_attention(output_rank)

            if self._output_shape:
                if not isinstance(self._output_shape, collections.abc.Sized):
                    output_shape = [self._output_shape]
                else:
                    output_shape = self._output_shape
            else:
                output_shape = [query_shape[-1]]

            # 构造 output dense
            # output 的时候要把 num_heads 合并
            # 因此 bound_dims 设置为 2
            einsum_equation, bias_axes, output_rank = _build_projection_equation(
                query_shape.rank - 1, bound_dims=2, output_dims=len(output_shape)
            )
            self._output_dense = EinsumDense(
                einsum_equation,
                output_shape=_get_output_shape(output_rank - 1, output_shape),
                bias_axes=bias_axes if self._use_bias else None,
                name="attention_output",
                **common_kwargs
            )

    def build_attention(self, rank):
        if self._attention_axes is None:
            # 第一个维度是 batch_size
            # 最后两个维度是 num_heads 和 hidden_size
            # 只有 seq_len 是做 attention 维度
            self._attention_axes = tuple(range(1, rank - 2))
        else:
            self._attention_axes = tuple(self._attention_axes)

        self._dot_product_equation, self._combine_equation, attention_scores_rank = _build_attention_equation(
            rank, attention_axes=self._attention_axes
        )
        norm_axes = tuple(
            range(attention_scores_rank - len(self._attention_axes), attention_scores_rank)
        )
        self._masked_softmax = masked_softmax_layer.MaskedSoftmax(
            mask_expansion_axes=[1], normalization_axes=norm_axes
        )
        self._dropout_layer = tf.keras.layers.Dropout(rate=self._attention_dropout_rate)

    def compute_attention(self, query, key, value, training, attention_mask=None):
        # do scale
        # [batch_size, seq_len, num_heads, size_per_head]
        query = tf.multiply(query, 1.0 / math.sqrt(float(self._size_per_head_for_query_and_key)))

        # dot-product
        # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = tf.einsum(self._dot_product_equation, query, key)

        # mask and softmax
        # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = self._masked_softmax(attention_scores, attention_mask)

        # dropout
        if training:
            attention_scores = self._dropout_layer(attention_scores)

        # combine heads and multi value
        attention_output = tf.einsum(self._combine_equation, attention_scores, value)

        # return
        return attention_output, attention_scores

    def call(self, query, value, training, key=None, attention_mask=None):
        """
        :param query: (batch_size, seq_len_q, hidden_size_q)
        :param value: (batch_size, seq_len_v, hidden_size_v)
        :param key: (batch_size, seq_len_k, hidden_size_k) if not given, will use value
        :param attention_mask: (batch_size, seq_len_q or 1, seq_len_v)
        :return: [batch_size, seq_len_q, output_shape]
        """
        # 为了加快运算速度，这里使用了自定义的运算
        # 而不是使用的默认的 dot-product 例如 tf.matmul()
        # _build_from_signature 就是在构造自定义点积运算
        if not self._built_from_signature:
            self._build_from_signature(query=query, value=value, key=key)
        if key is None:
            key = value

        # (batch_size, seq_len_q, num_heads, size_per_head_for_query_and_key)
        query = self._query_dense(query)

        # (batch_size, seq_len_k, num_heads, size_per_head_for_query_and_key)
        key = self._key_dense(key)

        # (batch_size, seq_len_v, num_heads, size_per_head_for_value)
        value = self._value_dense(value)

        attention_output, attention_scores = self.compute_attention(
            query, key, value, training=training, attention_mask=attention_mask
        )
        attention_output = self._output_dense(attention_output)

        if self._return_attention_scores:
            return attention_output, attention_scores
        return attention_output


class CacheAttention(MultiHeadAttention):
    """
        用于自回归解码器的 Attention
        除了 cache 部分其余都和基类一致
        只有在 predict 的时候 decoder 才需要 cache
        而在 train 的时候将 cache 设置为 None, 与基类保持一致即可

        cache 样例：
            {
                'key': (batch_size, seq_len_key, num_heads, hidden_size),
                'value': (batch_size, seq_len_value, num_heads, hidden_size)
            }
        其中 decoded_len 会随着解码的进行不断增加
        初始情况下应该使用 tf.zeros 初始化，表示 [START]
    """
    def _update_cache(
            self,
            key,
            value,
            cache
    ):
        key = tf.concat([tf.cast(cache['key'], key.dtype), key], axis=1)
        value = tf.concat([tf.cast(cache['value'], value.dtype), value], axis=1)

        cache['key'] = key
        cache['value'] = value

        return key, value

    def call(
            self,
            query,
            value,
            training,
            key=None,
            attention_mask=None,
            cache=None
    ):
        if not self._built_from_signature:
            self._build_from_signature(query=query, value=value, key=key)
        if key is None:
            key = value

        query = self._query_dense(query)

        key = self._key_dense(key)

        value = self._value_dense(value)

        if cache:
            key, value = self._update_cache(key, value, cache)

        query = tf.multiply(query, 1.0 / math.sqrt(float(self._size_per_head_for_query_and_key)))

        attention_scores = tf.einsum(self._dot_product_equation, key, query)

        attention_scores = self._masked_softmax(attention_scores, attention_mask)

        if training:
            attention_scores = self._dropout_layer(attention_scores)

        attention_output = tf.einsum(self._combine_equation, attention_scores, value)

        attention_output = self._output_dense(attention_output)

        if self._return_attention_scores:
            return attention_output, attention_scores, cache
        return attention_output, cache
