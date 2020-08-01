# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from layers.embedding_layers import OnDeviceEmbedding, PositionEmbedding
from layers.transformer_layers import TransformerEncoderLayer
from layers.attention_layers import SelfAttentionMask
from layers import einsum_dense


class TransformerEncoderForBert(tf.keras.Model):
    def __init__(
            self,
            vocab_size,
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            seq_len=None,
            max_seq_len=512,
            type_vocab_size=16,
            filter_size=3072,
            activation='relu',
            dropout_rate=0.1,
            attention_dropout_rate=0.1,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            return_all_encoder_outputs=False,
            output_range=None,
            embedding_size=None,
            embedding_layer=None,
            **kwargs
    ):
        # 定义输入
        # (batch_size, seq_len)
        inputs_ids = tf.keras.layers.Input(
            shape=(None,), dtype=tf.int32, name='inputs_ids'
        )

        # (batch_size, seq_len)
        inputs_mask = tf.keras.layers.Input(
            shape=(None,), dtype=tf.int32, name='inputs_mask'
        )

        # (batch_size, seq_len)
        inputs_type_ids = tf.keras.layers.Input(
            shape=(None,), dtype=tf.int32, name='inputs_type_ids'
        )

        # 定义各层

        # embedding layer
        if embedding_size is None:
            embedding_size = hidden_size
        if embedding_layer is None:
            self._embedding_layer = OnDeviceEmbedding(
                vocab_size=vocab_size,
                hidden_size=embedding_size,
                initializer=initializer,
                name='word_embedding'
            )
        else:
            self._embedding_layer = embedding_layer

        # position encoding layer
        self._position_embedding_layer = PositionEmbedding(
            initializer=initializer,
            use_dynamic_slicing=True,
            max_seq_len=max_seq_len,
            name='position_embedding'
        )

        # type embedding
        self._type_embedding_layer = OnDeviceEmbedding(
            vocab_size=vocab_size,
            hidden_size=embedding_size,
            initializer=initializer,
            use_one_hot=True,
            name='type_embedding'
        )

        if embedding_size != hidden_size:
            self._embedding_projection = einsum_dense.EinsumDense(
                '...x,xy->...y',
                output_shape=hidden_size,
                bias_axes='y',
                kernel_initializer=initializer,
                name='embedding_projection'
            )

        self._transformer_encoder_layers = []
        for i in range(num_layers):
            if i == num_layers - 1 and output_range is not None:
                transformer_output_range = output_range
            else:
                transformer_output_range = None

            layer = TransformerEncoderLayer(
                num_heads=num_heads,
                filter_size=filter_size,
                filter_activation=activation,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                output_range=transformer_output_range,
                kernel_initializer=initializer,
                name='transformer/layer_%d' % i
            )
            self._transformer_encoder_layers.append(layer)

        # 定义运算
        word_embeddings = self._embedding_layer(inputs_ids)
        position_embeddings = self._position_embedding_layer(word_embeddings)
        type_embeddings = self._type_embedding_layer(inputs_type_ids)
        embeddings = tf.keras.layers.Add()(
            [word_embeddings, position_embeddings, type_embeddings]
        )
        embeddings = tf.keras.layers.LayerNormalization(
            name='embedding/layer_norm',
            axis=-1,
            epsilon=1e-12,
            dtype=tf.float32
        )(embeddings)
        embeddings = tf.keras.layers.Dropout(rate=dropout_rate)(embeddings)
        embeddings = self._embedding_projection(embeddings)

        data = embeddings
        attention_mask = SelfAttentionMask()([data, mask])
        encoder_outputs = []
        for layer in self._transformer_encoder_layers:
            data = layer([data, attention_mask])
            encoder_outputs.append(data)

        first_token_tensor = tf.keras.layers.Lambda(
            lambda x: tf.squeeze(x[:, 0:1, :], axis=1)
        )(encoder_outputs[-1])
        self._pooler_layer = tf.keras.layers.Dense(
            units=hidden_size,
            activation='tanh',
            kernel_initializer=initializer,
            name='pooler_transform'
        )
        cls_output = self._pooler_layer(first_token_tensor)

        if return_all_encoder_outputs:
            outputs = [encoder_outputs, cls_output]
        else:
            outputs = [encoder_outputs[-1], cls_output]
        
        super(TransformerEncoderForBert, self).__init__(
            inputs=[inputs_ids, inputs_mask, inputs_type_ids],
            outputs=outputs,
            **kwargs
        )

    def get_embedding_table(self):
        return self._embedding_layer.embeddings

    def get_embedding_layer(self):
        return self._embedding_layer

    def transformer_layers(self):
        return self._transformer_encoder_layers

    def pooler_layer(self):
        return self._pooler_layer
