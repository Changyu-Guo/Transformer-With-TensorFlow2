# -*- coding: utf - 8 -*-

from collections import defaultdict

BASE_PARAMS = defaultdict(
    # 设置参数的默认值为 None
    lambda: None,

    # 输入相关
    default_batch_size=2048,
    default_batch_size_tpu=32769,
    max_length=256,

    # 模型相关
    initializer_gain=1.0,
    vocab_size=33708,
    hidden_size=512,
    num_hidden_layers=6,
    num_heads=8,
    filter_size=2048,

    # Dropout 相关
    layer_postprocess_dropout=0.1,
    attention_dropout=0.1,
    relu_dropout=0.1,

    # 训练相关
    label_smoothing=0.1,
    learning_rate=2.0,
    learning_rate_decay_rate=1.0,
    learning_rate_warmup_steps=16000,

    # 优化器相关
    optimizer_adam_beta_1=0.9,
    optimizer_adam_beta_2=0.997,
    optimizer_adam_epsilon=1e-09,

    # 预测相关
    extra_decode_length=50,
    beam_size=4,
    alpha=0.6,

    # tpu 相关
    use_tpu=False,
    static_batch=False,
    allow_ffn_pad=True
)

BIG_PARAMS = BASE_PARAMS.copy()
BIG_PARAMS.update(
    default_batch_size=4096,
    default_batch_size_tpu=16384,
    hidden_size=1024,
    filter_size=4096,
    num_heads=16
)

BASE_MULTI_GPU_PARAMS = BASE_PARAMS.copy()
BASE_MULTI_GPU_PARAMS.update(
    learning_rate_warmup_steps=8000
)

BIG_MULTI_GPU_PARAMS = BIG_PARAMS.copy()
BIG_MULTI_GPU_PARAMS.update(
    layer_postprocess_dropout=0.3,
    learning_rate_warmup_steps=8000
)

TINY_PARAMS = BASE_PARAMS.copy()
TINY_PARAMS.update(
    default_batch_size=32,
    default_batch_size_tpu=1024,
    hidden_size=32,
    num_heads=4,
    filter_size=256,
)