# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from tensorflow.keras import layers


class Attention(layers.Layer):
    def __init__(self, hidden_size, num_heads, attention_dropout):
        if hidden_size % num_heads:
            raise ValueError(
                'Hidden size ({}) must be divisible by number of header({})'.format(hidden_size, num_heads)
            )

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout

    def build(self, input_shape):
        """
        :param input_shape: [batch_size, seq_len, hidden_size]
        :return:
        """
        size_per_head = self.hidden_size // self.num_heads