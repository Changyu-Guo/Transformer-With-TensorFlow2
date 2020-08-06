# -*- coding: utf - 8 -*-

import tensorflow as tf
from layers import utils
from networks.encoders.bert_encoder import BertEncoder
from networks.encoders.albert_encoder import ALBertEncoder
from models.bert.configs import BertConfig
from models.albert.configs import ALBertConfig
from models.bert.bert_classifier import BertClassifier
from models.bert.bert_pretrainer import BertPretrainer


class BertPretrainLossAndMetricLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, **kwargs):
        super(BertPretrainLossAndMetricLayer, self).__init__(**kwargs)
        self._vocab_size = vocab_size
        self.config = {
            'vocab_size': vocab_size
        }

    def _add_metrics(
            self,
            lm_output,
            lm_labels,
            lm_labels_weights,
            lm_example_loss,
            sentence_output,
            sentence_labels,
            next_sentence_loss
    ):
        masked_lm_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
            lm_labels, lm_output
        )
        numerator = tf.reduce_sum(masked_lm_accuracy * lm_labels_weights)
        denominator = tf.reduce_sum(lm_labels_weights) + 1e-5
        masked_lm_accuracy = numerator / denominator
        self.add_metric(
            masked_lm_accuracy,
            name='masked_lm_accuracy',
            aggregation='mean'
        )
        self.add_metric(
            lm_example_loss,
            name='lm_example_loss',
            aggregation='mean'
        )

        if sentence_labels is not None:
            next_sentence_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
                sentence_labels, sentence_output
            )
            self.add_metric(
                next_sentence_accuracy,
                name='next_sentence_accuracy',
                aggregation='mean'
            )

        if next_sentence_loss is not None:
            self.add_metric(
                next_sentence_loss,
                name='next_sentence_loss',
                aggregation='mean'
            )

    def call(self, lm_output_logits, sentence_output_logits,
             lm_label_ids, lm_label_weights, sentence_labels=None):
        lm_label_weights = tf.cast(lm_label_weights, tf.float32)
        lm_output_logits = tf.cast(lm_output_logits, tf.float32)

        lm_prediction_losses = tf.keras.losses.sparse_categorical_crossentropy(
            lm_label_ids, lm_output_logits, from_logits=True
        )
        lm_numerator_loss = tf.reduce_sum(lm_prediction_losses * lm_label_weights)
        lm_denominator_loss = tf.reduce_sum(lm_label_weights)
        mask_label_loss = tf.math.divide_no_nan(lm_numerator_loss, lm_denominator_loss)

        if sentence_labels is not None:
            sentence_output_logits = tf.cast(sentence_output_logits, tf.float32)
            sentence_loss = tf.keras.losses.sparse_categorical_crossentropy(
                sentence_labels, sentence_output_logits, from_logits=True
            )
            sentence_loss = tf.reduce_mean(sentence_loss)
        else:
            sentence_loss = None
            loss = mask_label_loss

        batch_shape = tf.slice(tf.shape(lm_label_ids), [0], [1])
        final_loss = tf.fill(batch_shape, loss)
        self._add_metrics(
            lm_output_logits,
            lm_label_ids,
            lm_label_weights,
            mask_label_loss,
            sentence_output_logits,
            sentence_labels,
            sentence_loss
        )
        return final_loss


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


def create_pretrain_model(
        bert_config,
        seq_len,
        max_predictions_per_seq,
        initializer=None,
        use_next_sentence_label=True,
        return_core_pretrainer_model=False
):
    input_word_ids = tf.keras.layers.Input(
        shape=(seq_len,), name='input_word_ids', dtype=tf.int32
    )
    input_mask = tf.keras.layers.Input(
        shape=(seq_len,), name='input_mask', dtype=tf.int32
    )
    input_type_ids = tf.keras.layers.Input(
        shape=(seq_len,), name='input_type_ids', dtype=tf.int32
    )
    masked_lm_positions = tf.keras.layers.Input(
        shape=(max_predictions_per_seq,),
        name='masked_lm_positions',
        dtype=tf.int32
    )
    masked_lm_ids = tf.keras.layers.Input(
        shape=(max_predictions_per_seq,),
        name='masked_lm_ids',
        dtype=tf.int32
    )
    masked_lm_weights = tf.keras.layers.Input(
        shape=(max_predictions_per_seq,),
        name='masked_lm_weights',
        dtype=tf.int32
    )

    if use_next_sentence_label:
        next_sentence_labels = tf.keras.layers.Input(
            shape=(1,), name='next_sentence_labels', dtype=tf.int32
        )

    transformer_encoder = get_transformer_encoder(bert_config, seq_len)
    if initializer is None:
        initializer = tf.keras.initializers.TruncatedNormal(
            stddev=bert_config.initializer_range
        )
    pretrainer_model = BertPretrainer(
        network=transformer_encoder,
        embedding_table=transformer_encoder.get_embedding_table(),
        num_classes=2,
        activation=utils
    )

    outputs = pretrainer_model(
        [
            input_word_ids,
            input_mask,
            input_type_ids,
            masked_lm_positions
        ]
    )
    lm_output = outputs['masked_lm']
    sentence_output = outputs['classification']

    pretrain_loss_layer = BertPretrainLossAndMetricLayer(
        vocab_size=bert_config.vocab_size
    )
    output_loss = pretrain_loss_layer(
        lm_output,
        sentence_output,
        masked_lm_ids,
        masked_lm_weights,
        next_sentence_labels
    )

    inputs = {
        'input_word_ids': input_word_ids,
        'input_mask': input_mask,
        'input_type_ids': input_type_ids,
        'masked_lm_positions': masked_lm_positions,
        'masked_lm_ids': masked_lm_ids,
        'masked_lm_weights': masked_lm_weights
    }

    if use_next_sentence_label:
        inputs['next_sentence_labels'] = next_sentence_labels

    keras_model = tf.keras.Model(inputs=inputs, outputs=output_loss)
    if return_core_pretrainer_model:
        return keras_model, transformer_encoder, pretrainer_model
    else:
        return keras_model, transformer_encoder


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
