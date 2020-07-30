# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import model_params
import transformer

inputs = tf.random.uniform(shape=(2, 8), minval=0, maxval=41, dtype=tf.int64)
targets = tf.random.uniform(shape=(2, 4), minval=0, maxval=41, dtype=tf.int64)

params = model_params.TINY_PARAMS
params["batch_size"] = params["default_batch_size"] = 2
params["use_synthetic_data"] = True
params["hidden_size"] = 12
params["num_hidden_layers"] = 2
params["filter_size"] = 14
params["num_heads"] = 2
params["vocab_size"] = 41
params["extra_decode_length"] = 2
params["beam_size"] = 3
params["dtype"] = tf.float32

model = transformer.create_model(params, is_train=True)
model({
    'inputs': inputs,
    'targets': targets
})
