# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import json
import copy
import tensorflow as tf


class BertConfig:
    def __init__(
            self,
            vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=16,
            initializer_range=0.02,
            embedding_size=None,
            backward_compatible=True
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.embedding_size = embedding_size
        self.backward_compatible = backward_compatible

    @classmethod
    def from_dict(cls, json_obj):
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_obj):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_path):
        with tf.io.gfile.GFile(json_path, 'r') as f:
            json_str = f.read()
        return cls.from_dict(json.loads(json_str))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'


