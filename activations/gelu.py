# -*- coding: utf - 8 -*-

import numpy as np
import tensorflow as tf


def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
        np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))
    ))
    return x * cdf
