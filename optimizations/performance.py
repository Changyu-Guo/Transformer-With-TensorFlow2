# -*- coding: utf - 8 -*-

import tensorflow as tf


def set_mixed_precision_policy(dtype, loss_scale=None):
    if dtype == tf.float16:
        policy = tf.keras.mixed_precision.experimental.Policy(
            'mixed_float16', loss_scale=loss_scale
        )
        tf.keras.mixed_precision.experimental.set_policy(policy)
    elif dtype == tf.bfloat16:
        policy = tf.keras.mixed_precision.experimental.Policy(
            'mixed_bfloat16'
        )
        tf.keras.mixed_precision.experimental.set_policy(policy)
    elif dtype == tf.float32:
        tf.keras.mixed_precision.experimental.set_policy('float32')
    else:
        raise ValueError('Unexpected dtype: %s' % dtype)


def configure_optimizer(
        optimizer,
        use_float16=False,
        use_graph_rewrite=False,
        loss_scale='dynamic'
):
    if use_float16:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            optimizer, loss_scale=loss_scale
        )

    if use_graph_rewrite:
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
            optimizer
        )
        return optimizer
