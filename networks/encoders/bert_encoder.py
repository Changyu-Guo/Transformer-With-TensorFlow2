# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from layers.embedding_layers.word_embedding_layer import WordEmbedding
from layers.embedding_layers.bert_position_embedding_layer import BertPositionEmbedding
from layers.transformer_layers.encoder_layer import TransformerEncoderLayer
from layers.attention_layers.self_attention_mask import SelfAttentionMask
from layers.attention_layers.einsum_dense import EinsumDense
from activations.gelu import gelu


class BertEncoder(tf.keras.Model):
    def __init__(
            self,
            vocab_size,
            hidden_size=768,
            num_layers=12,
            num_attention_heads=12,
            max_seq_len=512,
            type_vocab_size=16,
            intermediate_size=3072,
            activation=gelu,
            hidden_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            return_all_encoder_outputs=False,
            output_range=None,
            embedding_size=None,
            embedding_layer=None,
            **kwargs
    ):
        activation = tf.keras.activations.get(activation)
        initializer = tf.keras.initializers.get(initializer)

        self._self_setattr_tracking = False
        self._config_dict = {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_attention_heads': num_attention_heads,
            'max_seq_len': max_seq_len,
            'type_vocab_size': type_vocab_size,
            'intermediate_size': intermediate_size,
            'activation': tf.keras.activations.serialize(activation),
            'hidden_dropout_rate': hidden_dropout_rate,
            'attention_dropout_rate': attention_dropout_rate,
            'initializer': tf.keras.initializers.serialize(initializer),
            'return_all_encoder_outputs': return_all_encoder_outputs,
            'output_range': output_range,
            'embedding_size': embedding_size
        }
        # 定义输入 words_ids, 类型为 int32
        # (batch_size, seq_len)
        inputs_ids = tf.keras.layers.Input(
            shape=(None,), dtype=tf.int32, name='inputs_ids'
        )

        # (batch_size, seq_len)
        # 类型为 int32
        inputs_mask = tf.keras.layers.Input(
            shape=(None,), dtype=tf.int32, name='inputs_mask'
        )

        # (batch_size, seq_len)
        # 类型为 int32
        inputs_type_ids = tf.keras.layers.Input(
            shape=(None,), dtype=tf.int32, name='inputs_type_ids'
        )

        # word embedding
        if embedding_size is None:
            embedding_size = hidden_size
        if embedding_layer is None:
            self._embedding_layer = WordEmbedding(
                vocab_size=vocab_size,
                embedding_size=embedding_size,
                initializer=initializer,
                name='word_embedding'
            )
        else:
            self._embedding_layer = embedding_layer

        # 经过 embedding 之后 dtype 会变成 float32
        word_embeddings = self._embedding_layer(inputs_ids)

        # position embedding
        self._position_embedding_layer = BertPositionEmbedding(
            initializer=initializer,
            use_dynamic_slicing=True,
            max_seq_len=max_seq_len,
            name='position_embedding'
        )
        position_embeddings = self._position_embedding_layer(word_embeddings)

        # type embedding
        self._type_embedding_layer = WordEmbedding(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            initializer=initializer,
            use_one_hot=True,
            name='type_embedding'
        )
        type_embeddings = self._type_embedding_layer(inputs_type_ids)

        # 将各种 embedding 加起来
        embeddings = tf.keras.layers.Add()(
            [word_embeddings, position_embeddings, type_embeddings]
        )

        # layer norm and dropout
        embeddings = tf.keras.layers.LayerNormalization(
            name='embedding/layer_norm',
            axis=-1,
            epsilon=1e-12,
            dtype=tf.float32
        )(embeddings)
        embeddings = tf.keras.layers.Dropout(rate=hidden_dropout_rate)(embeddings)

        # 将 embedding_size 投影到 hidden_size
        if embedding_size != hidden_size:
            self._embedding_projection = EinsumDense(
                '...x,xy->...y',
                output_shape=hidden_size,
                bias_axes='y',
                kernel_initializer=initializer,
                name='embedding_projection'
            )
            embeddings = self._embedding_projection(embeddings)

        self._transformer_encoder_layers = []
        data = embeddings
        attention_mask = SelfAttentionMask()([data, inputs_mask])
        encoder_outputs = []

        for i in range(num_layers):
            if i == num_layers - 1 and output_range is not None:
                transformer_output_range = output_range
            else:
                transformer_output_range = None

            layer = TransformerEncoderLayer(
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                intermediate_activation=activation,
                hidden_dropout_rate=hidden_dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                output_range=transformer_output_range,
                kernel_initializer=initializer,
                name='transformer/layer_%d' % i
            )
            self._transformer_encoder_layers.append(layer)
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
        # (batch_size, hidden_size)
        cls_output = self._pooler_layer(first_token_tensor)

        if return_all_encoder_outputs:
            outputs = [encoder_outputs, cls_output]
        else:
            outputs = [encoder_outputs[-1], cls_output]

        super(BertEncoder, self).__init__(
            inputs=[inputs_ids, inputs_mask, inputs_type_ids],
            outputs=outputs,
            **kwargs
        )

    def get_embedding_table(self):
        return self._embedding_layer.embeddings

    def get_embedding_layer(self):
        return self._embedding_layer

    def get_config(self):
        return self._config_dict

    @property
    def transformer_layers(self):
        return self._transformer_encoder_layers

    @property
    def pooler_layer(self):
        return self._pooler_layer

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
