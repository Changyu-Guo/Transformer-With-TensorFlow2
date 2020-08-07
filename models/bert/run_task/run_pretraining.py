# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.bert.create_bert_models import create_pretrain_model
from data_processors.bert_dataset_creators import create_pretrain_dataset


def get_pretrain_dataset_fn(
        input_file_pattern,
        seq_len,
        max_predictions_per_seq,
        global_batch_size,
        use_next_sentence_label=True
):
    def _dataset_fn(ctx=None):
        input_patterns = input_file_pattern.split(',')
        batch_size = ctx.get_per_replica_batch_size(global_batch_size)
        train_dataset = create_pretrain_dataset(
            input_patterns,
            seq_len,
            max_predictions_per_seq,
            batch_size,
            is_training=True,
            input_pipeline_context=ctx,
            use_next_sentence_label=use_next_sentence_label
        )
        return train_dataset

    return _dataset_fn


def get_loss_fn():
    def _bert_pretrain_loss_fn(unused_labels, losses, **unused_args):
        return tf.reduce_mean(losses)

    return _bert_pretrain_loss_fn


def run_customized_training(
        strategy,
        init_checkpoint,
        max_seq_len,
        max_predictions_per_seq,
        model_dir,
        steps_per_epoch,
        steps_per_loop,
        epochs,
        initial_lr,
        warmup_steps,
        end_lr,
        optimizer_type,
        input_files,
        train_batch_size,
        use_next_sentence_label=True,
        train_summary_interval=0,
        custom_callbacks=None
):
    train_input_fn = get_pretrain_dataset_fn(
        input_files,
        max_seq_len,
        max_predictions_per_seq,
        train_batch_size,
        use_next_sentence_label
    )

    def _get_pretrain_model():
        pretrain_model, core_model = create_pretrain_model(
            bert_config,
            max_seq_len,
            max_predictions_per_seq,
            use_next_sentence_label=use_next_sentence_label
        )
