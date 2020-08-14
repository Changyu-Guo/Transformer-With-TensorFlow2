# -*- coding: utf - 8 -*-

from collections import defaultdict

PARAMS = defaultdict(
    lambda: None,

    # input
    train_batch_size=2,
    eval_batch_size=2,
    max_seq_len=256,

    # model
    inputs_vocab_size=100,
    targets_vocab_size=100,
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=2048,
    intermediate_activation='gelu',
    use_bias=True,
    norm_first=False,
    norm_epsilon=0,

    # dropout
    hidden_dropout_rate=0.1,
    attention_dropout_rate=0.1,

    # training params
    label_smoothing=0.1,
    learning_rate=2.0,
    learning_rate_decay=1.0,
    learning_rate_warmup_steps=100,

    # optimizer adam
    optimizer_adam_beta_1=0.9,
    optimizer_adam_beta_2=0.997,
    optimizer_adam_epsilon=1e-9,

    # decode
    extra_decode_len=50,
    beam_size=4,
    alpha=0.6,

    # common
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)
