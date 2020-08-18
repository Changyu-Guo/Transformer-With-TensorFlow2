# -*- coding: utf - 8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
from absl import logging
import numpy as np
from tokenizations import sub_tokenization

"""
    1. 将待翻译的句子 tokenize 并转为 ids
    2. 在 ids 末尾添加 EOS
    3. 将 ids 输入模型，调用 predict 得到输入结果的 ids
    4. 将目标语言的 token ids 转为 tokens
"""

_EXTRA_DECODE_LENGTH = 100
_BEAM_SIZE = 4
_ALPHA = 0.6


def _trim_and_decode(ids, sub_tokenizer):
    try:
        index = list(ids).index(sub_tokenization.EOS_ID)
        return sub_tokenizer.decode(ids[:index])
    except ValueError:
        return sub_tokenizer.decode(ids)


def _encode_and_add_eos(line, sub_tokenizer):
    return sub_tokenizer.encode(line) + [sub_tokenization.EOS_ID]


def translate_from_text(model, sub_tokenizer, txt):
    encoded_txt = _encode_and_add_eos(txt, sub_tokenizer)
    result = model.predict(encoded_txt)
    outputs = result['outputs']
    logging.info('Original: "%s"' % txt)
    translate_from_input(outputs, sub_tokenizer)


def translate_from_input(outputs, sub_tokenizer):
    translation = _trim_and_decode(outputs, sub_tokenizer)
    logging.info('Translation: "%s"' % translation)


def _get_sorted_inputs(filename):
    with tf.io.gfile.GFile(filename) as f:
        records = f.read().split('\n')
        inputs = [record.strip() for record in records]
        if not inputs[-1]:
            inputs.pop()

    input_lens = [(i, len(line.split())) for i, line in enumerate(inputs)]
    sorted_input_lens = sorted(input_lens, key=lambda x: x[1], reverse=True)

    sorted_inputs = [None] * len(sorted_input_lens)
    sorted_keys = [0] * len(sorted_input_lens)
    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs[i] = inputs[index]
        sorted_inputs[index] = i
    return sorted_inputs, sorted_keys
