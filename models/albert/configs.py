# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from models.bert.configs import BertConfig


class ALBertConfig(BertConfig):
    def __init__(
            self,
            num_hidden_groups=1,
            inner_group_num=1,
            **kwargs
    ):
        super(ALBertConfig, self).__init__(**kwargs)

        if inner_group_num != 1 or num_hidden_groups != 1:
            raise ValueError(
                'Only support "inner_group_num" and '
                '"num_hidden_groups" as 1'
            )

    @classmethod
    def from_dict(cls, json_obj):
        config = ALBertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_obj):
            config.__dict__[key] = value
        return config
