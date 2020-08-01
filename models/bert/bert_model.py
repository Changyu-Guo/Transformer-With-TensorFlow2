# -*- coding: utf - 8 -*-

import tensorflow as tf
from layers import utils
from networks import transformer_encoder_for_bert, transformer_encoder_for_albert
from models.bert.configs import BertConfig, ALBertConfig
from models.bert.bert_classifier import BertClassifier


def get_transformer_encoder(
        bert_config,
        seq_len,
        transformer_encoder_cls=None,
        output_range=None
):
    if transformer_encoder_cls is not None:
        embedding_config = dict(
            vocab_size=bert_config.vocab_size,
            type_vocab_size=bert_config.type_vocab_size,
            hidden_size=bert_config.max_position_embeddings,
            initializer=tf.keras.initializers.TruncatedNormal(
                stddev=bert_config.initializer_range
            ),
            dropout_rate=bert_config.hidden_dropout_prob
        )
        hidden_config = dict(
            num_heads=bert_config.num_attention_heads,
            filter_size=bert_config.intermediate_size,
            filter_activation=None,
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
        num_attention_heads=bert_config.num_hidden_heads,
        filter_size=bert_config.intermediate_size,
        activation=None,
        dropout_rate=bert_config.hidden_dropout_prob,
        attention_dropout_rate=bert_config.attention_probs_dropout_prob,
        max_seq_len=bert_config.max_position_embeddings,
        type_vocab_size=bert_config.type_vocab_size,
        embedding_size=bert_config.embedding_size,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=bert_config.initializer_range
        )
    )

    if isinstance(bert_config, ALBertConfig):
        return transformer_encoder_for_albert.TransformerEncoderForALBert(**kwargs)
    else:
        assert isinstance(bert_config, BertConfig)
        kwargs['output_range'] = output_range
        return transformer_encoder_for_bert.TransformerEncoderForBert(**kwargs)


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
