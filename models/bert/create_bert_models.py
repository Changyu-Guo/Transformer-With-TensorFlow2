# -*- coding: utf - 8 -*-

import tensorflow as tf
from layers import utils
from networks.encoders.bert_encoder import BertEncoder
from networks.encoders.albert_encoder import ALBertEncoder
from models.bert.configs import BertConfig
from models.albert.configs import ALBertConfig
from models.bert.bert_classifier import BertClassifier


def get_transformer_encoder(
        bert_config,
        seq_len,
        transformer_encoder_cls=None,
        output_range=None
):
    del seq_len
    if transformer_encoder_cls is not None:
        embedding_config = dict(
            vocab_size=bert_config.vocab_size,
            type_vocab_size=bert_config.type_vocab_size,
            hidden_size=bert_config.hidden_size,
            max_seq_len=bert_config.max_position_embeddings,
            initializer=tf.keras.initializers.TruncatedNormal(
                stddev=bert_config.initializer_range
            ),
            dropout_rate=bert_config.hidden_dropout_prob
        )
        hidden_config = dict(
            num_attention_heads=bert_config.num_attention_heads,
            intermediate_size=bert_config.intermediate_size,
            intermediate_activation=tf.keras.activations.get(bert_config.hidden_act),
            dropout_rate=bert_config.hidden_dropout_prob,
            attention_dropout_rate=bert_config.attention_probs_dropout_prob,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=bert_config.initializer_range
            )
        )
        kwargs = dict(
            embedding_config=embedding_config,
            hidden_config=hidden_config,
            num_hidden_instances=bert_config.num_hidden_layers,
            pooled_output_dim=bert_config.hidden_size,
            pooler_layer_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=bert_config.initializer_range
            )
        )
        return transformer_encoder_cls(**kwargs)

    kwargs = dict(
        vocab_size=bert_config.vocab_size,
        hidden_size=bert_config.hidden_size,
        num_layers=bert_config.num_hidden_layers,
        num_attention_heads=bert_config.num_attention_heads,
        intermediate_size=bert_config.intermediate_size,
        activation=tf.keras.activations.get(bert_config.hidden_act),
        hidden_dropout_rate=bert_config.hidden_dropout_prob,
        attention_dropout_rate=bert_config.attention_probs_dropout_prob,
        max_seq_len=bert_config.max_position_embeddings,
        type_vocab_size=bert_config.type_vocab_size,
        embedding_size=bert_config.embedding_size,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=bert_config.initializer_range
        )
    )

    if isinstance(bert_config, ALBertConfig):
        return ALBertEncoder(**kwargs)
    else:
        assert isinstance(bert_config, BertConfig)
        kwargs['output_range'] = output_range
        return BertEncoder(**kwargs)


def create_classifier_model(
        bert_config,
        num_labels,
        max_seq_len=None,
        final_layer_initializer=None
):
    if final_layer_initializer is not None:
        initializer = final_layer_initializer
    else:
        initializer = tf.keras.initializers.TruncatedNormal(
            stddev=bert_config.initializer_range
        )

    bert_encoder = get_transformer_encoder(
        bert_config,
        max_seq_len,
        output_range=1
    )
    return BertClassifier(
        bert_encoder,
        num_classes=num_labels,
        dropout_rate=bert_config.hidden_dropout_prob,
        initializer=initializer
    ), bert_encoder
