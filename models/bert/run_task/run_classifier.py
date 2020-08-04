# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import json
import functools

import tensorflow as tf
from models.bert.create_bert_models import create_classifier_model


def get_loss_fn(num_classes):
    def classification_loss_fn(labels, logits):
        labels = tf.squeeze(labels)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(
            tf.cast(labels, dtype=tf.int32),
            depth=num_classes,
            dtype=tf.float32
        )
        per_example_loss = -tf.reduce_sum(
            tf.cast(one_hot_labels, dtype=tf.float32) * log_probs,
            axis=-1
        )
        return tf.reduce_mean(per_example_loss)

def run_bert_classifier(
        strategy,
        input_meta_data,
        model_dir,
        epochs,
        steps_per_epoch,
        steps_per_loop,
        eval_steps,
        warmup_steps,
        initial_lr,
        init_checkpoint,
        train_input_fn,
        eval_input_fn,
        training_callbacks=True,
        custom_callbacks=None,
        custom_metrics=None
):
    max_seq_len = input_meta_data['max_seq_len']
    num_classes = input_meta_data.get('num_labels', 1)
    is_regression = num_classes == 1

    def _get_classifier_model():
        classifier_model, core_model = create_classifier_model(
            bert_config,
            num_classes,
            max_seq_len
        )